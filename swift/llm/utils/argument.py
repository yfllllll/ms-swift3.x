# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Union

import json
import torch
import torch.distributed as dist
from torch import dtype as Dtype
from transformers.utils.versions import require_version

from swift import get_logger
from swift.hub import HubApi, ModelScopeConfig
from swift.utils import (add_version_to_work_dir, broadcast_string,
                         get_dist_setting, is_dist, is_master)
from .dataset import DATASET_MAPPING, get_custom_dataset, register_dataset
from .model import (MODEL_MAPPING, dtype_mapping,
                    get_default_lora_target_modules, get_default_template_type)
from .template import TEMPLATE_MAPPING, TemplateType
from .utils import is_vllm_available

logger = get_logger()


@dataclass
class SftArguments:
    # You can specify the model by either using the model_type or model_id_or_path.
    model_type: Optional[str] = field(
        default=None,
        metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    model_id_or_path: Optional[str] = None
    model_revision: Optional[str] = None
    model_cache_dir: Optional[str] = None

    sft_type: str = field(
        default='lora',
        metadata={'choices': ['lora', 'longlora', 'qalora', 'full']})
    freeze_parameters: float = 0.  # 0 ~ 1
    tuner_backend: str = field(
        default='swift', metadata={'choices': ['swift', 'peft']})
    template_type: str = field(
        default='AUTO',
        metadata={
            'help':
            f"template_type choices: {list(TEMPLATE_MAPPING.keys()) + ['AUTO']}"
        })
    output_dir: str = 'output'
    add_output_dir_suffix: bool = True
    custom_output_dir_suffix: str = None
    ddp_backend: str = field(
        default='nccl', metadata={'choices': ['nccl', 'gloo', 'mpi', 'ccl']})

    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    dtype: str = field(
        default='AUTO', metadata={'choices': ['bf16', 'fp16', 'fp32', 'AUTO']})

    dataset: Optional[List[str]] = field(
        default=None,
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_seed: int = 42
    dataset_test_ratio: float = 0.01
    train_dataset_sample: int = 20000  # -1: all dataset
    val_dataset_sample: Optional[int] = None  # -1: all dataset
    system: Optional[str] = None
    max_length: int = 2048  # -1: no limit
    truncation_strategy: str = field(
        default='delete', metadata={'choices': ['delete', 'truncation_left']})
    check_dataset_strategy: str = field(
        default='none',
        metadata={'choices': ['none', 'discard', 'error', 'warning']})
    custom_train_dataset_path: Optional[List[str]] = None
    custom_val_dataset_path: Optional[List[str]] = None
    self_cognition_sample: int = 0
    # Chinese name and English name
    model_name: Optional[List[str]] = None  # e.g. ['小黄', 'Xiao Huang']
    model_author: Optional[List[str]] = None  # e.g. ['魔搭', 'ModelScope']

    # If you want to use qlora, set the quantization_bit to 8 or 4.
    # And you need to install bitsandbytes: `pip install bitsandbytes -U`
    # note: bf16 and quantization have requirements for gpu architecture
    quantization_bit: int = field(default=0, metadata={'choices': [0, 4, 8]})
    bnb_4bit_comp_dtype: str = field(
        default='AUTO', metadata={'choices': ['fp16', 'bf16', 'fp32', 'AUTO']})
    bnb_4bit_quant_type: str = field(
        default='nf4', metadata={'choices': ['fp4', 'nf4']})
    bnb_4bit_use_double_quant: bool = True

    lora_target_modules: Optional[List[str]] = None
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout_p: float = 0.05

    neftune_alpha: float = 0.0

    gradient_checkpointing: Optional[bool] = None
    deepspeed_config_path: Optional[str] = None  # e.g. 'ds_config/zero2.json'
    batch_size: int = 1
    eval_batch_size: Optional[int] = None
    num_train_epochs: int = 1
    # if max_steps >= 0, override num_train_epochs
    max_steps: int = -1
    optim: str = 'adamw_torch'
    learning_rate: Optional[float] = None
    weight_decay: float = 0.01
    gradient_accumulation_steps: Optional[int] = None
    max_grad_norm: float = 0.5
    predict_with_generate: bool = False
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.05

    eval_steps: int = 50
    save_steps: Optional[int] = None
    only_save_model: Optional[bool] = None
    save_total_limit: int = 2  # save last and best. -1: all checkpoints
    logging_steps: int = 5
    dataloader_num_workers: int = 1

    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = True
    push_hub_strategy: str = field(
        default='push_best',
        metadata={
            'choices':
            ['end', 'push_best', 'push_last', 'checkpoint', 'all_checkpoints']
        })
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'SDK token can be found in https://modelscope.cn/my/myaccesstoken'
        })

    # other
    test_oom_error: bool = field(
        default=False,
        metadata={
            'help':
            'If set to True, the train_dataset will be sorted in descending order based on max_length, '
            'enabling faster detection of OOM (Out of Memory) errors.'
        })
    disable_tqdm: bool = False
    lazy_tokenize: bool = False
    preprocess_num_proc: int = 1
    use_flash_attn: Optional[bool] = None
    ignore_args_error: bool = False  # True: notebook compatibility
    logging_dir: Optional[str] = None
    report_to: Optional[List[str]] = None
    check_model_is_latest: bool = True
    acc_strategy: str = field(
        default='token', metadata={'choices': ['token', 'sentence']})
    save_on_each_node: bool = True
    save_strategy: str = field(
        default='steps', metadata={'choices': ['steps', 'no']})
    save_safetensors: bool = True

    # generation config
    max_new_tokens: int = 2048
    do_sample: bool = True
    temperature: float = 0.3
    top_k: int = 20
    top_p: float = 0.7
    repetition_penalty: float = 1.05

    def __post_init__(self) -> None:
        handle_compatibility(self)
        handle_path(self)
        set_model_type(self)
        register_custom_dataset(self)
        check_flash_attn(self)
        if self.self_cognition_sample > 0:
            if self.model_name is None or self.model_author is None:
                raise ValueError(
                    'Please enter self.model_name self.model_author. '
                    'For example: `--model_name 小黄 "Xiao Huang" --model_author 魔搭 ModelScope`. '
                    'Representing the model name and model author in Chinese and English.'
                )
            for k in ['model_name', 'model_author']:
                v = getattr(self, k)
                if len(v) == 1:
                    v = v[0]
                if isinstance(v, str):
                    setattr(self, k, [v, v])
            if self.sft_type == 'lora' and 'ALL' not in self.lora_target_modules:
                logger.warning(
                    'Due to knowledge editing involved, it is recommended to add LoRA on MLP. '
                    'For example: `--lora_target_modules ALL`. '
                    'If you have already added LoRA on MLP, please ignore this warning.'
                )

        self.torch_dtype, self.fp16, self.bf16 = select_dtype(self)
        world_size = 1
        if is_dist():
            rank, local_rank, world_size, _ = get_dist_setting()
            torch.cuda.set_device(local_rank)
            self.seed += rank  # Avoid the same dropout
            if self.ddp_backend == 'gloo' and self.quantization_bit != 0:
                raise ValueError('not supported, please use `nccl`')

            # Initialize in advance
            if not dist.is_initialized():
                dist.init_process_group(backend=self.ddp_backend)

        if self.add_output_dir_suffix:
            self.output_dir = os.path.join(self.output_dir, self.model_type)
            if self.custom_output_dir_suffix is not None:
                self.output_dir = os.path.join(self.output_dir,
                                               self.custom_output_dir_suffix)
            else:
                self.output_dir = add_version_to_work_dir(self.output_dir)
            logger.info(f'output_dir: {self.output_dir}')

        if self.sft_type in ('lora', 'longlora', 'qalora'):
            assert self.freeze_parameters == 0., (
                'lora does not support `freeze_parameters`, please set `--sft_type full`'
            )
            if 'int4' in self.model_type or 'int8' in self.model_type:
                assert self.quantization_bit == 0
            if self.learning_rate is None:
                self.learning_rate = 1e-4
            if self.only_save_model is None:
                if self.deepspeed_config_path is None:
                    self.only_save_model = False
                else:
                    self.only_save_model = True
        elif self.sft_type == 'full':
            assert 0 <= self.freeze_parameters < 1
            assert self.quantization_bit == 0, 'not supported'
            assert self.dtype != 'fp16', 'please use bf16 or fp32'
            if self.learning_rate is None:
                self.learning_rate = 2e-5
            if self.only_save_model is None:
                self.only_save_model = True
        else:
            raise ValueError(f'sft_type: {self.sft_type}')

        if self.template_type == 'AUTO':
            self.template_type = get_default_template_type(self.model_type)
            logger.info(f'Setting template_type: {self.template_type}')
        if isinstance(self.dataset, str):
            self.dataset = [self.dataset]
        elif self.dataset is None:
            self.dataset = []
        if len(self.dataset) == 0 and (len(self.custom_train_dataset_path) == 0
                                       and len(
                                           self.custom_val_dataset_path) == 0
                                       and self.self_cognition_sample == 0):
            raise ValueError(f'self.dataset: {self.dataset}')

        if self.save_steps is None:
            self.save_steps = self.eval_steps
        if self.lora_target_modules is None:
            self.lora_target_modules = ['DEFAULT']
        elif isinstance(self.lora_target_modules, str):
            self.lora_target_modules = [self.lora_target_modules]
        if 'DEFAULT' in self.lora_target_modules or 'AUTO' in self.lora_target_modules:
            assert len(self.lora_target_modules) == 1
            self.lora_target_modules = get_default_lora_target_modules(
                self.model_type)
        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = select_bnb(
            self)

        prepare_push_ms_hub(self)
        self.train_sampler_random = not self.test_oom_error
        if self.eval_batch_size is None:
            if self.predict_with_generate:
                self.eval_batch_size = 1
            else:
                self.eval_batch_size = self.batch_size
        if self.save_total_limit == -1:
            self.save_total_limit = None
        if self.max_length == -1:
            self.max_length = None

        self.deepspeed = None
        if self.deepspeed_config_path is not None:
            require_version('deepspeed')
            with open(self.deepspeed_config_path, 'r', encoding='utf-8') as f:
                self.deepspeed = json.load(f)
            logger.info(f'Using deepspeed: {self.deepspeed}')
        if self.logging_dir is None:
            self.logging_dir = f'{self.output_dir}/runs'
        if self.report_to is None:
            self.report_to = ['all']
        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = math.ceil(16 / self.batch_size
                                                         / world_size)
        if self.preprocess_num_proc > 1 and 'qwen-audio' in self.model_type:
            logger.warning(
                'qwen-audio does not support multi-process preprocess, '
                'because qwen-audio\'s preprocessing functions use torch\'s multi-processing, '
                'which causes compatibility issues. '
                'You can use `--lazy_tokenize true` to avoid long preprocessing times.'
            )
            self.preprocess_num_proc = 1
            logger.info(
                f'Setting self.preprocess_num_proc: {self.preprocess_num_proc}'
            )
        model_info = MODEL_MAPPING[self.model_type]
        support_gradient_checkpointing = model_info.get(
            'support_gradient_checkpointing', True)
        if self.gradient_checkpointing is None:
            self.gradient_checkpointing = support_gradient_checkpointing
        elif not support_gradient_checkpointing:
            assert self.gradient_checkpointing is False, 'not support gradient_checkpointing'


@dataclass
class InferArguments:
    # You can specify the model by either using the model_type or model_id_or_path.
    model_type: Optional[str] = field(
        default=None,
        metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    model_id_or_path: Optional[str] = None
    model_revision: Optional[str] = None
    model_cache_dir: Optional[str] = None

    sft_type: str = field(
        default='lora',
        metadata={'choices': ['lora', 'longlora', 'qalora', 'full']})
    template_type: str = field(
        default='AUTO',
        metadata={
            'help':
            f"template_type choices: {list(TEMPLATE_MAPPING.keys()) + ['AUTO']}"
        })
    infer_backend: str = field(
        default='AUTO', metadata={'choices': ['AUTO', 'vllm', 'pt']})
    ckpt_dir: Optional[str] = field(
        default=None, metadata={'help': '/path/to/your/vx_xxx/checkpoint-xxx'})
    load_args_from_ckpt_dir: bool = True
    load_dataset_config: bool = False
    eval_human: Optional[bool] = None

    seed: int = 42
    dtype: str = field(
        default='AUTO', metadata={'choices': ['bf16', 'fp16', 'fp32', 'AUTO']})

    dataset: Optional[List[str]] = field(
        default=None,
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_seed: int = 42
    dataset_test_ratio: float = 0.01
    val_dataset_sample: int = 10  # -1: all dataset
    save_result: bool = True
    system: Optional[str] = None
    max_length: int = 2048  # -1: no limit
    truncation_strategy: str = field(
        default='delete', metadata={'choices': ['delete', 'truncation_left']})
    check_dataset_strategy: str = field(
        default='none',
        metadata={'choices': ['none', 'discard', 'error', 'warning']})
    custom_train_dataset_path: Optional[List[str]] = None
    custom_val_dataset_path: Optional[List[str]] = None

    quantization_bit: int = field(default=0, metadata={'choices': [0, 4, 8]})
    bnb_4bit_comp_dtype: str = field(
        default='AUTO', metadata={'choices': ['fp16', 'bf16', 'fp32', 'AUTO']})
    bnb_4bit_quant_type: str = field(
        default='nf4', metadata={'choices': ['fp4', 'nf4']})
    bnb_4bit_use_double_quant: bool = True

    max_new_tokens: int = 2048
    do_sample: bool = True
    temperature: float = 0.3
    top_k: int = 20
    top_p: float = 0.7
    repetition_penalty: float = 1.05

    # other
    use_flash_attn: Optional[bool] = None
    ignore_args_error: bool = False  # True: notebook compatibility
    stream: bool = True
    merge_lora_and_save: bool = False
    save_safetensors: bool = True
    overwrite_generation_config: bool = False
    verbose: Optional[bool] = None
    # app-ui
    share: bool = False
    # vllm
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    # compatibility. (Deprecated parameter.)
    show_dataset_sample: int = 10
    safe_serialization: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.ckpt_dir is not None and not self.check_ckpt_dir_correct(
                self.ckpt_dir):
            raise ValueError(
                f'The checkpoint dir {self.ckpt_dir} passed in is invalid, please make sure'
                'the dir contains a `configuration.json` file.')
        handle_compatibility(self)
        handle_path(self)
        logger.info(f'ckpt_dir: {self.ckpt_dir}')
        if self.ckpt_dir is None and self.load_args_from_ckpt_dir:
            self.load_args_from_ckpt_dir = False
            logger.info(
                'Due to `ckpt_dir` being `None`, `load_args_from_ckpt_dir` is set to `False`.'
            )
        if self.load_args_from_ckpt_dir:
            load_from_ckpt_dir(self)
        else:
            assert self.load_dataset_config is False
            set_model_type(self)
        register_custom_dataset(self)
        check_flash_attn(self)

        self.torch_dtype, _, _ = select_dtype(self)
        if self.template_type == 'AUTO':
            self.template_type = get_default_template_type(self.model_type)
            logger.info(f'Setting template_type: {self.template_type}')
        if isinstance(self.dataset, str):
            self.dataset = [self.dataset]
        elif self.dataset is None:
            self.dataset = []
        has_dataset = (
            len(self.dataset) > 0 or len(self.custom_train_dataset_path) > 0
            or len(self.custom_val_dataset_path) > 0)
        if self.eval_human is None:
            if not has_dataset:
                self.eval_human = True
            else:
                self.eval_human = False
            logger.info(f'Setting self.eval_human: {self.eval_human}')
        elif self.eval_human is False and not has_dataset:
            raise ValueError(
                'Please provide the dataset or set `--load_dataset_config true`.'
            )
        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = select_bnb(
            self)

        if self.max_length == -1:
            self.max_length = None
        if self.ckpt_dir is None and self.overwrite_generation_config:
            self.overwrite_generation_config = False
            logger.warning('Setting overwrite_generation_config: False')
        if self.ckpt_dir is None:
            self.sft_type = 'full'
        model_info = MODEL_MAPPING[self.model_type]
        support_vllm = model_info.get('support_vllm', False)
        if self.infer_backend == 'AUTO':
            if is_vllm_available() and support_vllm:
                if (self.sft_type == 'full'
                        or self.sft_type == 'lora' and self.merge_lora_and_save
                        and self.quantization_bit == 0):
                    self.infer_backend = 'vllm'
                else:
                    self.infer_backend = 'pt'
        if self.infer_backend == 'vllm':
            assert self.quantization_bit == 0, 'not support bnb'
            assert support_vllm, f'vllm not support `{self.model_type}`'
            if self.sft_type == 'lora':
                assert self.merge_lora_and_save is True, 'please set `--merge_lora_and_save true`'

    @staticmethod
    def check_ckpt_dir_correct(ckpt_dir) -> bool:
        """Check the checkpoint dir is correct, which means it must contains a `configuration.json` file.
        Args:
            ckpt_dir: The checkpoint dir
        Returns:
            A bool value represents the dir is valid or not.
        """
        if not os.path.exists(ckpt_dir):
            return False
        return os.path.isfile(os.path.join(ckpt_dir, 'configuration.json'))


@dataclass
class DPOArguments(SftArguments):

    ref_model_type: Optional[str] = field(
        default=None,
        metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})

    max_prompt_length: int = 1024


@dataclass
class RomeArguments(InferArguments):
    rome_request_file: str = field(
        default=None,
        metadata={
            'help':
            'The rome request file, please check the documentation '
            'to get the format'
        })

    def __post_init__(self) -> None:
        handle_compatibility(self)
        handle_path(self)
        set_model_type(self)
        check_flash_attn(self)

        self.torch_dtype, _, _ = select_dtype(self)
        if self.template_type == 'AUTO':
            self.template_type = get_default_template_type(self.model_type)
            logger.info(f'Setting template_type: {self.template_type}')

        if self.max_length == -1:
            self.max_length = None


dtype_mapping_reversed = {v: k for k, v in dtype_mapping.items()}


def select_dtype(
        args: Union[SftArguments, InferArguments]) -> Tuple[Dtype, bool, bool]:
    if not torch.cuda.is_available():
        if args.dtype == 'AUTO':
            args.dtype = 'fp32'
            logger.info(f'Setting args.dtype: {args.dtype}')
        assert args.dtype != 'fp16', 'The CPU does not support matrix multiplication with FP16.'
        if args.dtype == 'fp32':
            return torch.float32, False, False
        elif args.dtype == 'bf16':
            return torch.bfloat16, False, True
        else:
            raise ValueError(f'args.dtype: {args.dtype}')

    if args.dtype == 'AUTO' and not torch.cuda.is_bf16_supported():
        args.dtype = 'fp16'
    if args.dtype == 'AUTO' and ('int4' in args.model_type
                                 or 'int8' in args.model_type):
        model_torch_dtype = MODEL_MAPPING[args.model_type]['torch_dtype']
        if model_torch_dtype is not None:
            args.dtype = dtype_mapping[model_torch_dtype]
    if args.dtype == 'AUTO':
        args.dtype = 'bf16'

    torch_dtype = dtype_mapping_reversed[args.dtype]

    assert torch_dtype in {torch.float16, torch.bfloat16, torch.float32}
    if torch_dtype == torch.float16:
        if isinstance(args, SftArguments) and args.sft_type == 'full':
            args.dtype = 'fp32'
            torch_dtype = torch.float32
            logger.warning(
                'Fine-tuning with full parameters does not support fp16, and is prone to NaN. '
                'We will use the fp32 & AMP approach, which consumes approximately twice the memory of bf16.'
            )
            logger.info(f'Setting torch_dtype: {torch_dtype}')
        fp16, bf16 = True, False
    elif torch_dtype == torch.bfloat16:
        support_bf16 = torch.cuda.is_bf16_supported()
        if not support_bf16:
            logger.warning(f'support_bf16: {support_bf16}')
        fp16, bf16 = False, True
    else:
        fp16, bf16 = False, False
    return torch_dtype, fp16, bf16


def select_bnb(
        args: Union[SftArguments, InferArguments]) -> Tuple[Dtype, bool, bool]:
    if args.bnb_4bit_comp_dtype == 'AUTO':
        args.bnb_4bit_comp_dtype = args.dtype

    quantization_bit = args.quantization_bit
    bnb_4bit_compute_dtype = dtype_mapping_reversed[args.bnb_4bit_comp_dtype]
    assert bnb_4bit_compute_dtype in {
        torch.float16, torch.bfloat16, torch.float32
    }
    if quantization_bit == 4:
        require_version('bitsandbytes')
        load_in_4bit, load_in_8bit = True, False
    elif quantization_bit == 8:
        require_version('bitsandbytes')
        load_in_4bit, load_in_8bit = False, True
    else:
        load_in_4bit, load_in_8bit = False, False

    return bnb_4bit_compute_dtype, load_in_4bit, load_in_8bit


def handle_compatibility(args: Union[SftArguments, InferArguments]) -> None:
    if args.dataset is not None and len(
            args.dataset) == 1 and ',' in args.dataset[0]:
        args.dataset = args.dataset[0].split(',')
    if args.template_type == 'chatglm2-generation':
        args.template_type = 'chatglm-generation'
    if args.template_type == 'qwen':
        args.template_type = TemplateType.chatml
    if (isinstance(args, InferArguments) and args.show_dataset_sample != 10
            and args.val_dataset_sample == 10):
        # args.val_dataset_sample is the default value and args.show_dataset_sample is not the default value.
        args.val_dataset_sample = args.show_dataset_sample
    if args.truncation_strategy == 'ignore':
        args.truncation_strategy = 'delete'
    if isinstance(args,
                  InferArguments) and args.safe_serialization is not None:
        args.save_safetensors = args.safe_serialization


def set_model_type(args: Union[SftArguments, InferArguments]) -> None:
    assert args.model_type is None or args.model_id_or_path is None, (
        '`model_type` and `model_id_or_path` can only specify one of them.')
    if args.model_id_or_path is not None:
        model_mapping_reversed = {
            v['model_id_or_path'].lower(): k
            for k, v in MODEL_MAPPING.items()
        }
        model_id_or_path = args.model_id_or_path
        model_id_or_path_lower = model_id_or_path.lower()
        if model_id_or_path_lower not in model_mapping_reversed:
            if isinstance(args,
                          InferArguments) and 'checkpoint' in model_id_or_path:
                error_msg = 'Please use `--ckpt_dir vx_xxx/checkpoint-xxx` to use the checkpoint.'
            else:
                error_msg = f"`model_id_or_path`: '{model_id_or_path}' is not registered."
                if os.path.exists(model_id_or_path):
                    error_msg += (
                        ' Please use `--model_id_or_path <model_id> --model_cache_dir <local_path>` '
                        'to specify the local cache path for the model.')
            raise ValueError(error_msg)
        args.model_type = model_mapping_reversed[model_id_or_path_lower]

    if args.model_type is None:
        raise ValueError(
            'please setting `--model_type <model_type>` or `--model_id_or_path <model_id>`'
        )
    elif args.model_type not in MODEL_MAPPING:
        raise ValueError(f'model_type: {args.model_type} is not registered.')
    model_info = MODEL_MAPPING[args.model_type]
    if args.model_revision is None:
        args.model_revision = model_info['revision']
    else:
        model_info['revision'] = args.model_revision
    args.model_id_or_path = model_info['model_id_or_path']
    requires = model_info['requires']
    for require in requires:
        require_version(require)


def prepare_push_ms_hub(args: SftArguments) -> None:
    if args.hub_model_id is None:
        args.hub_model_id = f'{args.model_type}-{args.sft_type}'
        logger.info(f'Setting hub_model_id: {args.hub_model_id}')
    if args.push_to_hub:
        api = HubApi()
        if args.hub_token is None:
            args.hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
        if args.hub_token is not None:
            api.login(args.hub_token)
        else:
            assert ModelScopeConfig.get_token(
            ) is not None, 'Please enter hub_token'
        logger.info('hub login successful!')


def _check_path(
        k: str, value: Union[str, List[str]],
        check_exist_path_set: Optional[Set[str]]) -> Union[str, List[str]]:
    if isinstance(value, str):
        value = os.path.expanduser(value)
        value = os.path.abspath(value)
        if k in check_exist_path_set and not os.path.exists(value):
            raise FileNotFoundError(f"`{k}`: '{value}'")
    elif isinstance(value, list):
        res = []
        for v in value:
            res.append(_check_path(k, v, check_exist_path_set))
        value = res
    return value


def handle_path(args: Union[SftArguments, InferArguments]) -> None:
    check_exist_path = [
        'model_cache_dir', 'ckpt_dir', 'resume_from_checkpoint',
        'deepspeed_config_path', 'custom_train_dataset_path',
        'custom_val_dataset_path'
    ]
    if args.model_id_or_path is not None and (
            args.model_id_or_path.startswith('~')
            or args.model_id_or_path.startswith('/')):
        check_exist_path.append('model_id_or_path')
    check_exist_path_set = set(check_exist_path)
    other_path = ['output_dir', 'logging_dir']
    for k in check_exist_path + other_path:
        value = getattr(args, k, None)
        if value is None:
            continue
        value = _check_path(k, value, check_exist_path_set)
        setattr(args, k, value)


def register_custom_dataset(args: Union[SftArguments, InferArguments]) -> None:
    for key in ['custom_train_dataset_path', 'custom_val_dataset_path']:
        value = getattr(args, key)
        if value is None:
            setattr(args, key, [])
        elif isinstance(value, str):
            setattr(args, key, [value])
    if len(args.custom_train_dataset_path) == 0 and len(
            args.custom_val_dataset_path) == 0:
        return
    register_dataset(
        '_custom_dataset',
        '_custom_dataset',
        args.custom_train_dataset_path,
        args.custom_val_dataset_path,
        get_function=get_custom_dataset,
        exists_ok=True)
    if args.dataset is None:
        args.dataset = ['_custom_dataset']
    elif '_custom_dataset' not in args.dataset:
        args.dataset.append('_custom_dataset')


def load_from_ckpt_dir(args: InferArguments) -> None:
    sft_args_path = os.path.join(args.ckpt_dir, 'sft_args.json')
    if not os.path.exists(sft_args_path):
        logger.info(f'{sft_args_path} not found')
        return
    with open(sft_args_path, 'r', encoding='utf-8') as f:
        sft_args = json.load(f)
    imported_keys = [
        'model_type', 'model_id_or_path', 'model_revision', 'sft_type',
        'template_type', 'dtype', 'system', 'quantization_bit',
        'bnb_4bit_comp_dtype', 'bnb_4bit_quant_type',
        'bnb_4bit_use_double_quant'
    ]
    if args.load_dataset_config:
        imported_keys += [
            'dataset', 'dataset_seed', 'dataset_test_ratio',
            'check_dataset_strategy', 'custom_train_dataset_path',
            'custom_val_dataset_path'
        ]
    for key in imported_keys:
        if (key in {
                'dataset', 'custom_train_dataset_path',
                'custom_val_dataset_path'
        } and getattr(args, key) is not None):
            continue
        setattr(args, key, sft_args.get(key))


def check_flash_attn(args: Union[SftArguments, InferArguments]) -> None:
    model_info = MODEL_MAPPING[args.model_type]
    support_flash_attn = model_info.get('support_flash_attn', False)
    if args.use_flash_attn and not support_flash_attn:
        logger.warning(f'use_flash_attn: {args.use_flash_attn}, '
                       f'but support_flash_attn: {support_flash_attn}')
