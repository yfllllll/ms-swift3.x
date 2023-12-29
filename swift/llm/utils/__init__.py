# Copyright (c) Alibaba, Inc. and its affiliates.
from .argument import DPOArguments, InferArguments, RomeArguments, SftArguments
from .dataset import (DATASET_MAPPING, DatasetName, GetDatasetFunction,
                      HfDataset, add_self_cognition_dataset, get_dataset,
                      get_dataset_from_repo, load_dataset_from_local,
                      load_ms_dataset, register_dataset)
from .model import (MODEL_MAPPING, GetModelTokenizerFunction, LoRATM,
                    ModelType, get_additional_saved_files,
                    get_default_lora_target_modules, get_default_template_type,
                    get_model_tokenizer, get_model_tokenizer_from_repo,
                    get_model_tokenizer_with_flash_attn, register_model)
from .preprocess import (AlpacaPreprocessor, ClsPreprocessor,
                         ComposePreprocessor, ConversationsPreprocessor,
                         PreprocessFunc, RenameColumnsPreprocessor,
                         SmartPreprocessor, SwiftPreprocessor,
                         TextGenerationPreprocessor)
from .template import (DEFAULT_SYSTEM, TEMPLATE_MAPPING, History, Prompt,
                       Template, TemplateType, get_template, register_template)
from .utils import (LazyLLMDataset, LLMDataset, data_collate_fn, dataset_map,
                    download_dataset, find_all_linear_for_lora,
                    fix_fp16_trainable_bug, history_to_messages, inference,
                    inference_stream, is_vllm_available, limit_history_length,
                    messages_to_history, print_example, set_generation_config,
                    sort_by_max_length, stat_dataset)

try:
    if is_vllm_available():
        from .vllm_utils import (VllmGenerationConfig, get_vllm_engine,
                                 inference_stream_vllm, inference_vllm,
                                 prepare_vllm_engine_template)
except Exception as e:
    from swift.utils import get_logger
    logger = get_logger()
    logger.warning(f'import vllm_utils error: {e}')
