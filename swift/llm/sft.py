# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial
from typing import Any, Dict, Union

import json
import numpy as np
import torch
from modelscope import BitsAndBytesConfig, GenerationConfig
from transformers import IntervalStrategy
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import is_torch_npu_available

from swift.trainers import Seq2SeqTrainer
from swift.utils import (check_json_format, compute_acc_metrics,
                         compute_nlg_metrics, get_dist_setting, get_logger,
                         get_main, get_model_info, is_ddp_plus_mp, is_dist,
                         is_master, plot_images, preprocess_logits_for_metrics,
                         seed_everything, show_layers)
from .tuner import prepare_model
from .utils import (TEMPLATE_MAPPING, LazyLLMDataset, SftArguments, Template,
                    add_self_cognition_dataset, dataset_map, get_dataset,
                    get_model_tokenizer, get_template, get_time_info,
                    print_example, set_generation_config, sort_by_max_length,
                    stat_dataset)
from .utils.argument import handle_dataset_mixture

logger = get_logger()


def llm_sft(args: SftArguments) -> Dict[str, Union[str, Any]]:
    logger.info(f'args: {args}')
    training_args = args.training_args
    if is_torch_npu_available():
        print(f'device_count: {torch.npu.device_count()}')
    else:
        print(f'device_count: {torch.cuda.device_count()}')
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    print(f'rank: {rank}, local_rank: {local_rank}, '
          f'world_size: {world_size}, local_world_size: {local_world_size}')
    seed_everything(args.seed)

    if args.gpu_memory_fraction:
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(
                max(min(args.gpu_memory_fraction, 1.0), 0.01),
                device=device_id)

    # Loading Model and Tokenizer
    if is_torch_npu_available():
        model_kwargs = {'device_map': local_rank if local_rank >= 0 else 0}
    elif is_deepspeed_zero3_enabled():
        model_kwargs = {'device_map': None}
    else:
        model_kwargs = {'low_cpu_mem_usage': True}
        if is_dist() and not is_ddp_plus_mp():
            model_kwargs['device_map'] = {'': local_rank}
        else:
            model_kwargs['device_map'] = 'auto'
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        model_kwargs['quantization_config'] = quantization_config

    kwargs = {}
    if args.use_flash_attn is not None:
        kwargs['use_flash_attn'] = args.use_flash_attn
    model, tokenizer = get_model_tokenizer(
        args.model_type,
        args.torch_dtype,
        model_kwargs,
        model_id_or_path=args.model_id_or_path,
        is_training=True,
        **kwargs)
    # logger.info(f'device_map: {dict(model.hf_device_map)}')
    logger.info(f'model_config: {model.config}')
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    logger.info(f'generation_config: {generation_config}')
    set_generation_config(model, generation_config)
    training_args.generation_config = generation_config

    # Preparing LoRA
    model, callbacks = prepare_model(model, args)

    show_layers(model)
    model_info = None
    if not is_deepspeed_zero3_enabled():
        model_info = get_model_info(model)
        logger.info(model_info)
    logger.info(model)

    if args.gradient_checkpointing:
        model.config.use_cache = False  # fix transformers==4.36
        logger.info('Setting model.config.use_cache: False')
        model.enable_input_require_grads()

    # Loading Dataset
    random_state = np.random.RandomState(args.dataset_seed)
    train_dataset, val_dataset = get_dataset(
        args.dataset,
        args.dataset_test_ratio,
        random_state,
        check_dataset_strategy=args.check_dataset_strategy)
    val_dataset_sample = args.val_dataset_sample
    mix_dataset_sample = 0 if not args.train_dataset_mix_ratio else round(
        len(train_dataset) * args.train_dataset_mix_ratio)
    if train_dataset is not None and args.train_dataset_sample >= 0:
        total_dataset_sample = min(args.train_dataset_sample,
                                   train_dataset.shape[0])
        train_dataset_sample = total_dataset_sample
        if args.train_dataset_mix_ratio:
            train_dataset_sample = round(
                1. / (1 + args.train_dataset_mix_ratio) * total_dataset_sample)
            mix_dataset_sample = total_dataset_sample - train_dataset_sample
        if train_dataset.shape[0] > train_dataset_sample:
            logger.info(f'train_dataset_sample: {train_dataset_sample}')
            train_idxs = random_state.permutation(train_dataset_sample)
            train_dataset = train_dataset.select(train_idxs)
        if val_dataset_sample is None:
            val_dataset_sample = max(
                int(train_dataset_sample * args.dataset_test_ratio), 1)
    if val_dataset is not None and val_dataset_sample is not None and val_dataset_sample >= 0:
        if val_dataset.shape[0] > val_dataset_sample:
            logger.info(f'val_dataset_sample: {val_dataset_sample}')
            val_idxs = random_state.permutation(val_dataset_sample)
            val_dataset = val_dataset.select(val_idxs)

    train_dataset = handle_dataset_mixture(args, train_dataset,
                                           mix_dataset_sample)

    # add self-cognition dataset
    if args.self_cognition_sample > 0:
        train_dataset = add_self_cognition_dataset(train_dataset,
                                                   args.self_cognition_sample,
                                                   args.model_name,
                                                   args.model_author)
    if val_dataset is None:
        training_args.evaluation_strategy = IntervalStrategy.NO
        training_args.do_eval = False
    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    template_kwargs = {}
    template_info = TEMPLATE_MAPPING[args.template_type]
    use_model = template_info.get('use_model', False)
    if use_model:
        template_kwargs['model'] = model
    template_kwargs['use_loss_scale'] = args.use_loss_scale
    template: Template = get_template(args.template_type, tokenizer,
                                      args.system, args.max_length,
                                      args.truncation_strategy,
                                      **template_kwargs)
    args.system = template.default_system
    logger.info(f'system: {args.system}')
    logger.info(f'args.lazy_tokenize: {args.lazy_tokenize}')
    if not args.lazy_tokenize:
        dataset_info = {}
        logger.info(f'Using num_proc: {args.preprocess_num_proc}')
        train_dataset = dataset_map(train_dataset, template.encode,
                                    args.preprocess_num_proc)
        if val_dataset is not None:
            val_dataset = dataset_map(val_dataset, template.encode,
                                      args.preprocess_num_proc)
        if args.test_oom_error:
            train_dataset = sort_by_max_length(train_dataset, 20000)
        # Data analysis
        td0, tkwargs0 = train_dataset.data[0]
        print_example(td0, tokenizer, tkwargs0)
        dataset_info['train_dataset'] = stat_dataset(train_dataset)
        if val_dataset is not None:
            dataset_info['val_dataset'] = stat_dataset(val_dataset)
    else:
        dataset_info = None
        td0, tkwargs0 = template.encode(train_dataset[0])
        print_example(td0, tokenizer, tkwargs0)
        train_dataset = LazyLLMDataset(train_dataset, template)
        if val_dataset is not None:
            val_dataset = LazyLLMDataset(val_dataset, template)

    padding_to = args.max_length if args.sft_type == 'longlora' else None
    data_collator = partial(template.data_collator, padding_to=padding_to)

    # Trainer
    logger.info(f'training_args: {training_args}')

    trainer_kwargs = {}
    if args.predict_with_generate:
        trainer_kwargs['compute_metrics'] = partial(
            compute_nlg_metrics, tokenizer=tokenizer)
    else:
        compute_metrics = partial(
            compute_acc_metrics, acc_strategy=args.acc_strategy)
        trainer_kwargs['compute_metrics'] = compute_metrics
        trainer_kwargs[
            'preprocess_logits_for_metrics'] = preprocess_logits_for_metrics
    if args.check_model_is_latest is False:
        trainer_kwargs['check_model'] = False

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
        **trainer_kwargs)
    trainer.sft_args = args
    if is_master():
        for args_obj, fname in zip([args, training_args],
                                   ['sft_args.json', 'training_args.json']):
            fpath = os.path.join(args.output_dir, fname)
            logger.info(
                f'The {args_obj.__class__.__name__} will be saved in: {fpath}')
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(
                    check_json_format(args_obj.__dict__),
                    f,
                    ensure_ascii=False,
                    indent=2)
    logging_path = os.path.join(args.output_dir, 'logging.jsonl')
    logger.info(f'The logging file will be saved in: {logging_path}')
    trainer.train(training_args.resume_from_checkpoint)
    last_model_checkpoint = getattr(trainer.state, 'last_model_checkpoint',
                                    None)
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
    logger.info(
        f'best_model_checkpoint: {trainer.state.best_model_checkpoint}')
    train_time = get_time_info(trainer.state.log_history, len(train_dataset))
    # Visualization
    if is_master():
        images_dir = os.path.join(args.output_dir, 'images')
        logger.info(f'images_dir: {images_dir}')
        plot_images(images_dir, args.logging_dir, ['train/loss'], 0.9)
        if args.push_to_hub:
            trainer._add_patterns_to_gitignore(['images/'])
            trainer.push_to_hub()
    return {
        'memory': trainer.perf['memory'],
        'train_time': train_time,
        'last_model_checkpoint': last_model_checkpoint,
        'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'best_metric': trainer.state.best_metric,
        'global_step': trainer.state.global_step,
        'log_history': trainer.state.log_history,
        'model_info': model_info,
        'dataset_info': dataset_info,
    }


sft_main = get_main(SftArguments, llm_sft)
