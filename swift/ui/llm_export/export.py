from typing import Type

import gradio as gr

from swift.llm import DATASET_MAPPING
from swift.ui.base import BaseUI


class Export(BaseUI):

    group = 'llm_export'

    locale_dict = {
        'merge_lora': {
            'label': {
                'zh': '合并lora',
                'en': 'Merge lora'
            },
        },
        'quant_bits': {
            'label': {
                'zh': '量化比特数',
                'en': 'Quantize bits'
            },
        },
        'quant_method': {
            'label': {
                'zh': '量化方法',
                'en': 'Quantize method'
            },
        },
        'quant_n_samples': {
            'label': {
                'zh': '量化集采样数',
                'en': 'Sampled rows from calibration dataset'
            },
        },
        'quant_seqlen': {
            'label': {
                'zh': '量化集的max-length',
                'en': 'The quantize sequence length'
            },
        },
        'quant_output_dir': {
            'label': {
                'zh': '量化输出路径',
                'en': 'Output dir for quantization'
            },
        },
        'dataset': {
            'label': {
                'zh': '校准数据集',
                'en': 'Calibration datasets'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Row():
            gr.Checkbox(elem_id='merge_lora', scale=20)
            gr.Textbox(elem_id='quant_bits', scale=20)
            gr.Dropdown(elem_id='quant_method', scale=20)
            gr.Textbox(elem_id='quant_n_samples', scale=20)
            gr.Textbox(elem_id='quant_seqlen', scale=20)
        with gr.Row():
            gr.Textbox(elem_id='quant_output_dir', value='output', scale=20)
            gr.Dropdown(elem_id='dataset', multiselect=True, choices=list(DATASET_MAPPING.keys()), scale=20)
