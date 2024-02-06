# Copyright (c) Alibaba, Inc. and its affiliates.
import types

import torch
import transformers
from packaging import version

from swift.trainers import TrainerCallback
from swift.tuners import (AdaLoraConfig, IA3Config, LongLoRAConfig,
                          LongLoRAModelType, LoraConfig, LoRAConfig,
                          NEFTuneConfig, Swift)
from swift.utils import (activate_model_parameters, freeze_model_parameters,
                         get_logger)
from .utils import SftArguments, find_all_linears, find_embedding, is_adapter

logger = get_logger()


def handle_target_modules_all(model, args: SftArguments) -> None:
    if args.sft_type == 'ia3':
        target_modules = args.ia3_target_modules
        assert len(args.ia3_feedforward_modules) > 0, (
            'Setting ia3_target_modules to `ALL` '
            'need to pass MLP linear names to `ia3_feedforward_modules`')
    else:
        target_modules = args.lora_target_modules
    if args.lora_use_embedding:
        target_modules += find_embedding(model)
    if args.lora_use_all:
        target_modules += find_all_linears(model, args.quantization_bit,
                                           args.model_type)
    if args.sft_type == 'ia3':
        args.ia3_target_modules = target_modules
        logger.info(f'ia3_target_modules: {args.ia3_target_modules}')
    else:
        args.lora_target_modules = target_modules
        logger.info(f'lora_target_modules: {args.lora_target_modules}')


def prepare_model(model, args: SftArguments):
    # Preparing LoRA
    if is_adapter(args.sft_type):
        if args.resume_from_checkpoint is None:
            handle_target_modules_all(model, args)
            lora_kwargs = {
                'r': args.lora_rank,
                'target_modules': args.lora_target_modules,
                'lora_alpha': args.lora_alpha,
                'lora_dropout': args.lora_dropout_p,
                'bias': args.lora_bias_trainable,
                'modules_to_save': args.lora_modules_to_save,
                'layers_to_transform': args.lora_layers_to_transform,
                'layers_pattern': args.lora_layers_pattern,
                'rank_pattern': args.lora_rank_pattern,
                'alpha_pattern': args.lora_alpha_pattern,
                'loftq_config': args.lora_loftq_config,
            }
            if args.sft_type == 'lora':
                if args.tuner_backend == 'swift':
                    lora_config = LoRAConfig(
                        lora_dtype=args.lora_dtype, **lora_kwargs)
                elif args.tuner_backend == 'peft':
                    lora_config = LoraConfig(
                        task_type='CAUSAL_LM', **lora_kwargs)
                model = Swift.prepare_model(model, lora_config)
                logger.info(f'lora_config: {lora_config}')
            elif args.sft_type == 'longlora':
                assert args.tuner_backend == 'swift'
                assert LongLoRAModelType.LLAMA in args.model_type
                longlora_config = LongLoRAConfig(
                    lora_dtype=args.lora_dtype,
                    model_type=LongLoRAModelType.LLAMA,
                    use_flash_attn=args.use_flash_attn,
                    **lora_kwargs)
                model = Swift.prepare_model(model, longlora_config)
                logger.info(f'longlora_config: {longlora_config}')
            elif args.sft_type == 'qalora':
                assert args.tuner_backend == 'swift'
                assert getattr(
                    model, 'quantization_method',
                    None) == 'gptq', 'qalora must be used with auto_gptq'
                qalora_config = LoRAConfig(
                    lora_dtype=args.lora_dtype,
                    use_qa_lora=True,
                    **lora_kwargs)
                model = Swift.prepare_model(model, qalora_config)
                logger.info(f'qalora_config: {qalora_config}')
            elif args.sft_type == 'adalora':
                lora_kwargs['rank_pattern'] = None
                adalora_config = AdaLoraConfig(
                    task_type='CAUSAL_LM',
                    **lora_kwargs,
                    target_r=args.adalora_target_r,
                    init_r=args.adalora_init_r,
                    tinit=args.adalora_tinit,
                    tfinal=args.adalora_tfinal,
                    deltaT=args.adalora_deltaT,
                    beta1=args.adalora_beta1,
                    beta2=args.adalora_beta2,
                    orth_reg_weight=args.adalora_orth_reg_weight,
                )
                model = Swift.prepare_model(model, adalora_config)
                logger.info(f'adalora_config: {adalora_config}')
            elif args.sft_type == 'ia3':
                ia3_config = IA3Config(
                    task_type='CAUSAL_LM',
                    target_modules=args.ia3_target_modules,
                    feedforward_modules=args.ia3_feedforward_modules or [],
                    modules_to_save=args.ia3_modules_to_save,
                )
                model = Swift.prepare_model(model, ia3_config)
                logger.info(f'ia3_config: {ia3_config}')
        else:
            model = Swift.from_pretrained(
                model, args.resume_from_checkpoint, is_trainable=True)
        # fix bug: Attempting to unscale FP16 gradients.
        #   peft: https://github.com/huggingface/peft/issues/1249
        #   modules_to_save + fp16
        is_logging = False
        for p in model.parameters():
            if p.requires_grad and p.dtype == torch.float16:
                if not is_logging:
                    logger.info(
                        'Convert trainable parameters from fp16 to fp32.')
                    is_logging = True
                p.data = p.data.to(dtype=torch.float32)
    elif args.sft_type == 'full':
        if args.freeze_parameters > 0:
            freeze_model_parameters(model, args.freeze_parameters)
        if len(args.additional_trainable_parameters) > 0:
            activate_model_parameters(model,
                                      args.additional_trainable_parameters)
    else:
        raise ValueError(f'args.sft_type: {args.sft_type}')

    if version.parse(transformers.__version__) < version.parse(
            '4.35') and args.neftune_noise_alpha not in {None, 0.}:
        neftune_config = NEFTuneConfig(noise_alpha=args.neftune_noise_alpha)
        model = Swift.prepare_model(model, {'neftune': neftune_config})
        logger.info(f'neftune_config: {neftune_config}')

    class TrainerAdapterCallback(TrainerCallback):

        def __init__(self):
            self.global_step = 0

        # offload original_modules to cpu, to save memory
        def on_train_begin(self, _args, state, control, **kwargs):
            if hasattr(model, 'set_active_adapters'):
                model.set_active_adapters(model.adapters.keys(), offload='cpu')
            if args.sft_type == 'adalora':
                model.peft_config['default'].total_step = state.max_steps

                def zero_grad(_self, *args, **kwargs):
                    _self.update_and_allocate(self.global_step + 1)
                    _self._zero_grad(*args, **kwargs)

                model._zero_grad = model.zero_grad
                model.zero_grad = types.MethodType(zero_grad, model)

        def on_step_end(self, _args, state, control, **kwargs):
            if args.sft_type == 'adalora':
                self.global_step = state.global_step

    callbacks = []
    if is_adapter(args.sft_type) and args.tuner_backend == 'swift':
        callbacks.append(TrainerAdapterCallback())
    return model, callbacks
