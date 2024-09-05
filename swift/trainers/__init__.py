# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .arguments import Seq2SeqTrainingArguments, TrainingArguments
    from .dpo_trainer import DPOTrainer
    from .orpo_trainer import ORPOTrainer
    from .rlhf_trainers import RLHFTrainerFactory
    from .trainers import Seq2SeqTrainer, Trainer
    from .loss import LOSS_MAPPING, LossName, register_loss_func, get_loss_func
    from .utils import (EvaluationStrategy, FSDPOption, HPSearchBackend, HubStrategy, IntervalStrategy, SchedulerType,
                        ShardedDDPOption, TrainerCallback, concat_template, sort_by_max_length,
                        get_preprocessed_rlhf_dataset, patch_trl)
else:
    _import_structure = {
        'arguments': ['Seq2SeqTrainingArguments', 'TrainingArguments'],
        'dpo_trainer': ['DPOTrainer'],
        'orpo_trainer': ['ORPOTrainer'],
        'rlhf_trainers': ['RLHFTrainerFactory'],
        'trainers': ['Seq2SeqTrainer', 'Trainer'],
        'loss': ['LOSS_MAPPING', 'LossName', 'register_loss_func', 'get_loss_func'],
        'utils': [
            'EvaluationStrategy', 'FSDPOption', 'HPSearchBackend', 'HubStrategy', 'IntervalStrategy', 'SchedulerType',
            'ShardedDDPOption', 'TrainerCallback', 'concat_template', 'get_preprocessed_rlhf_dataset', 'patch_trl'
        ]
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
