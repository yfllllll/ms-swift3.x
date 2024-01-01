import os
import sys
import time
from dataclasses import fields
from typing import Dict, Type

import gradio as gr
import json
import torch
from gradio import Accordion, Tab

from swift.llm import SftArguments
from swift.ui.base import BaseUI
from swift.ui.llm_train.advanced import Advanced
from swift.ui.llm_train.dataset import Dataset
from swift.ui.llm_train.hyper import Hyper
from swift.ui.llm_train.lora import LoRA
from swift.ui.llm_train.model import Model
from swift.ui.llm_train.quantization import Quantization
from swift.ui.llm_train.runtime import Runtime
from swift.ui.llm_train.save import Save
from swift.ui.llm_train.self_cog import SelfCog
from swift.utils import get_logger

logger = get_logger()


class LLMTrain(BaseUI):

    group = 'llm_train'

    sub_ui = [
        Model,
        Dataset,
        Runtime,
        Save,
        LoRA,
        Hyper,
        Quantization,
        SelfCog,
        Advanced,
    ]

    locale_dict: Dict[str, Dict] = {
        'llm_train': {
            'label': {
                'zh': 'LLM训练',
                'en': 'LLM Training',
            }
        },
        'submit_alert': {
            'value': {
                'zh':
                '任务已开始，请查看tensorboard或日志记录，关闭本页面不影响训练过程',
                'en':
                'Task started, please check the tensorboard or log file, '
                'closing this page does not affect training'
            }
        },
        'submit': {
            'value': {
                'zh': '🚀 开始训练',
                'en': '🚀 Begin'
            }
        },
        'dry_run': {
            'label': {
                'zh': '仅生成运行命令',
                'en': 'Dry-run'
            },
            'info': {
                'zh': '仅生成运行命令，开发者自行运行',
                'en': 'Generate run command only, for manually running'
            }
        },
        'gpu_id': {
            'label': {
                'zh': '选择可用GPU',
                'en': 'Choose GPU'
            },
            'info': {
                'zh': '选择训练使用的GPU号，如CUDA不可用只能选择CPU',
                'en': 'Select GPU to train'
            }
        },
        'gpu_memory_fraction': {
            'label': {
                'zh': 'GPU显存限制',
                'en': 'GPU memory fraction'
            },
            'info': {
                'zh':
                '设置使用显存的比例，一般用于显存测试',
                'en':
                'Set the memory fraction ratio of GPU, usually used in memory test'
            }
        },
        'sft_type': {
            'label': {
                'zh': '训练方式',
                'en': 'Train type'
            },
            'info': {
                'zh': '选择训练的方式',
                'en': 'Select the training type'
            }
        },
        'seed': {
            'label': {
                'zh': '随机数种子',
                'en': 'Seed'
            },
            'info': {
                'zh': '选择随机数种子',
                'en': 'Select a random seed'
            }
        },
        'dtype': {
            'label': {
                'zh': '训练精度',
                'en': 'Training Precision'
            },
            'info': {
                'zh': '选择训练精度',
                'en': 'Select the training precision'
            }
        },
        'use_ddp': {
            'label': {
                'zh': '使用DDP',
                'en': 'Use DDP'
            },
            'info': {
                'zh': '是否使用数据并行训练',
                'en': 'Use Distributed Data Parallel to train'
            }
        },
        'neftune_alpha': {
            'label': {
                'zh': 'neftune_alpha',
                'en': 'neftune_alpha'
            },
            'info': {
                'zh': '使用neftune提升训练效果',
                'en': 'Use neftune to improve performance'
            }
        }
    }

    choice_dict = {}
    default_dict = {}
    for f in fields(SftArguments):
        if 'choices' in f.metadata:
            choice_dict[f.name] = f.metadata['choices']
        default_dict[f.name] = getattr(SftArguments, f.name)

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_train', label=''):
            gpu_count = 0
            default_device = 'cpu'
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                default_device = '0'
            with gr.Blocks():
                Model.build_ui(base_tab)
                Dataset.build_ui(base_tab)
                Runtime.build_ui(base_tab)
                with gr.Row():
                    gr.Dropdown(elem_id='sft_type', scale=4)
                    gr.Textbox(elem_id='seed', scale=4)
                    gr.Dropdown(elem_id='dtype', scale=4)
                    gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
                    gr.Slider(
                        elem_id='neftune_alpha',
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        scale=4)
                with gr.Row():
                    gr.Dropdown(
                        elem_id='gpu_id',
                        multiselect=True,
                        choices=[str(i) for i in range(gpu_count)] + ['cpu'],
                        value=default_device,
                        scale=8)
                    gr.Textbox(
                        elem_id='gpu_memory_fraction', value='1.0', scale=4)
                    gr.Checkbox(elem_id='dry_run', value=False, scale=4)
                    submit = gr.Button(
                        elem_id='submit', scale=4, variant='primary')

                Save.build_ui(base_tab)
                LoRA.build_ui(base_tab)
                Hyper.build_ui(base_tab)
                Quantization.build_ui(base_tab)
                SelfCog.build_ui(base_tab)
                Advanced.build_ui(base_tab)
                submit.click(
                    cls.train, [
                        value for value in cls.elements().values()
                        if not isinstance(value, (Tab, Accordion))
                    ], [
                        cls.element('running_cmd'),
                        cls.element('logging_dir'),
                        cls.element('runtime_tab')
                    ],
                    show_progress=True)

    @classmethod
    def train(cls, *args):
        ignore_elements = ('model_type', 'logging_dir', 'more_params')
        sft_args = fields(SftArguments)
        sft_args = {
            arg.name: getattr(SftArguments, arg.name)
            for arg in sft_args
        }
        kwargs = {}
        kwargs_is_list = {}
        other_kwargs = {}
        more_params = {}
        keys = [
            key for key, value in cls.elements().items()
            if not isinstance(value, (Tab, Accordion))
        ]
        for key, value in zip(keys, args):
            compare_value = sft_args.get(key)
            compare_value_arg = str(compare_value) if not isinstance(
                compare_value, (list, dict)) else compare_value
            compare_value_ui = str(value) if not isinstance(
                value, (list, dict)) else value
            if key not in ignore_elements and key in sft_args and compare_value_ui != compare_value_arg and value:
                kwargs[key] = value if not isinstance(
                    value, list) else ' '.join(value)
                kwargs_is_list[key] = isinstance(value, list) or getattr(
                    cls.element(key), 'is_list', False)
            else:
                other_kwargs[key] = value
            if key == 'more_params' and value:
                more_params = json.loads(value)

        kwargs.update(more_params)
        sft_args = SftArguments(**kwargs)
        params = ''

        for e in kwargs:
            if kwargs_is_list[e]:
                params += f'--{e} {kwargs[e]} '
            else:
                params += f'--{e} "{kwargs[e]}" '
        params += '--add_output_dir_suffix False '
        for key, param in more_params.items():
            params += f'--{key} "{param}" '
        ddp_param = ''
        devices = other_kwargs['gpu_id']
        devices = [d for d in devices if d]
        if other_kwargs['use_ddp']:
            ddp_param = f'NPROC_PER_NODE={len(devices)}'
        assert (len(devices) == 1 or 'cpu' not in devices)
        gpus = ','.join(devices)
        cuda_param = ''
        if gpus != 'cpu':
            cuda_param = f'CUDA_VISIBLE_DEVICES={gpus}'

        log_file = os.path.join(sft_args.logging_dir, 'run.log')
        if sys.platform == 'win32':
            if cuda_param:
                cuda_param = f'set {cuda_param} && '
            if ddp_param:
                ddp_param = f'set {ddp_param} && '
            run_command = f'{cuda_param}{ddp_param}start /b swift sft {params} > {log_file} 2>&1'
        else:
            run_command = f'{cuda_param} {ddp_param} nohup swift sft {params} > {log_file} 2>&1 &'
        logger.info(f'Run training: {run_command}')
        if not other_kwargs['dry_run']:
            os.makedirs(sft_args.logging_dir, exist_ok=True)
            os.system(run_command)
            time.sleep(1)  # to make sure the log file has been created.
            gr.Info(cls.locale('submit_alert', cls.lang)['value'])
        return run_command, sft_args.logging_dir, gr.update(visible=True)
