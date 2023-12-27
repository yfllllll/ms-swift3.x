# Copyright (c) Alibaba, Inc. and its affiliates.

from .adapter import Adapter, AdapterConfig
from .longlora.longlora import LongLoRA, LongLoRAConfig
from .lora import LoRA, LoRAConfig
from .neftune import NEFTune, NEFTuneConfig
from .prompt import Prompt, PromptConfig
from .restuning import ResTuning, ResTuningConfig
from .rome import Rome, RomeConfig
from .scetuning.scetuning import SCETuning, SCETuningConfig
from .side import Side, SideConfig


class SwiftTuners:
    ADAPTER = 'ADAPTER'
    PROMPT = 'PROMPT'
    LORA = 'LORA'
    SIDE = 'SIDE'
    RESTUNING = 'RESTUNING'
    ROME = 'ROME'
    LONGLORA = 'longlora'
    NEFTUNE = 'neftune'
    SCETUNING = 'SCETuning'


SWIFT_MAPPING = {
    SwiftTuners.ADAPTER: (AdapterConfig, Adapter),
    SwiftTuners.PROMPT: (PromptConfig, Prompt),
    SwiftTuners.LORA: (LoRAConfig, LoRA),
    SwiftTuners.SIDE: (SideConfig, Side),
    SwiftTuners.RESTUNING: (ResTuningConfig, ResTuning),
    SwiftTuners.ROME: (RomeConfig, Rome),
    SwiftTuners.LONGLORA: (LongLoRAConfig, LongLoRA),
    SwiftTuners.NEFTUNE: (NEFTuneConfig, NEFTune),
    SwiftTuners.SCETUNING: (SCETuningConfig, SCETuning),
}
