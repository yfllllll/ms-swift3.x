# Copyright (c) Alibaba, Inc. and its affiliates.
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase, StoppingCriteria

DEFAULT_SYSTEM = 'You are a helpful assistant.'  # qwen system
History = List[Union[Tuple[str, str], List[str]]]


class TemplateType:
    # text-generation
    default_generation = 'default-generation'
    default_generation_bos = 'default-generation-bos'
    chatglm_generation = 'chatglm-generation'
    # chat
    default = 'default'
    chatml = 'chatml'
    qwen = chatml
    baichuan = 'baichuan'
    chatglm2 = 'chatglm2'
    chatglm3 = 'chatglm3'
    llama = 'llama'
    openbuddy = 'openbuddy'
    internlm = 'internlm'
    yi = 'yi'
    xverse = 'xverse'
    ziya = 'ziya'
    skywork = 'skywork'
    bluelm = 'bluelm'
    zephyr = 'zephyr'


Prompt = List[Union[str, List[Union[str, int]]]]
StopWords = Prompt

Context = Union[str, List[int]]


def _simplify_context_list(
        context_list: List[Context],
        compute_loss_idx: List[int]) -> Tuple[List[Context], List[int]]:
    res: List[Context] = []
    res_idx: List[int] = []
    temp: List[str] = []
    compute_loss_idx = set(compute_loss_idx)
    for i, c in enumerate(context_list):
        if isinstance(c, str) and i not in compute_loss_idx:
            temp.append(c)
        else:
            if len(temp) > 0:
                res.append(''.join(temp))
                temp.clear()
            res.append(c)
            if i in compute_loss_idx:
                res_idx.append(len(res) - 1)
    if len(temp) > 0:
        res.append(''.join(temp))
    return res, res_idx


def get_audio_info(
        tokenizer: PreTrainedTokenizerBase,
        *,
        context: Optional[str] = None,
        audio_info: Optional[Dict[str,
                                  Any]] = None) -> Optional[Dict[str, Any]]:
    assert context is not None or audio_info is not None
    assert context is None or audio_info is None
    if context is None:
        input_audios = audio_info.get('input_audios')
        if isinstance(input_audios, Tensor):
            return audio_info
        audio_urls = audio_info['audio_urls']
        context = ''.join([f'<audio>{url}</audio>' for url in audio_urls])
    return tokenizer.process_audio(context)


def _concat_context_list(
    context_list: List[Context],
    res_context_list: List[Context],
    compute_loss_idx: List[int],
    system: Optional[str] = None,
    query: Optional[str] = None,
    response: Optional[str] = None,
    round0: Optional[int] = None,
) -> None:
    # concat context list and replace placeholder
    round1 = None
    if round0 is not None:
        round1 = str(round0 + 1)
        round0 = str(round0)
    for context in context_list:
        if isinstance(context, str):
            if '{{RESPONSE}}' == context:
                assert response is not None
                res_context_list.append(response)
                compute_loss_idx.append(len(res_context_list) - 1)
                continue
            old_str_list = [
                '{{SYSTEM}}', '{{QUERY}}', '{{ROUND0}}', '{{ROUND1}}'
            ]
            new_str_list = [system, query, round0, round1]
            for (old_str, new_str) in zip(old_str_list, new_str_list):
                if new_str is not None and old_str in context:
                    context = context.replace(old_str, new_str)
        res_context_list.append(context)


def _encode_context_list(
    tokenizer: PreTrainedTokenizerBase,
    context_list: List[Context],
    compute_loss_idx: Optional[List[int]] = None
) -> Tuple[List[int], Optional[List[int]], Dict[str, Any]]:
    input_ids: List[int] = []
    labels: List[int] = []
    kwargs = {}
    if compute_loss_idx is not None:
        compute_loss_idx = set(compute_loss_idx)
    for i, context in enumerate(context_list):
        if isinstance(context, list):
            for c in context:
                if isinstance(c, str):
                    token = getattr(tokenizer, c)
                    assert token is not None
                else:
                    token = c
                input_ids.append(token)
                labels.append(-100)
        elif isinstance(context, str):
            if (getattr(tokenizer, 'model_type', '').startswith('qwen-audio')):
                audio_info = get_audio_info(tokenizer, context=context)
                old_audio_info = kwargs.get('audio_info')
                if old_audio_info is None:
                    kwargs['audio_info'] = audio_info
                elif audio_info is not None:
                    for k in ['input_audios', 'input_audio_lengths']:
                        old_audio_info[k] = torch.concat(
                            [old_audio_info[k], audio_info[k]], dim=0)
                    for k in ['audio_span_tokens', 'audio_urls']:
                        old_audio_info[k] = old_audio_info[k] + audio_info[k]
            token_list = tokenizer(
                context,
                return_attention_mask=False,
                add_special_tokens=False,
                **kwargs)['input_ids']
            input_ids += token_list
            if compute_loss_idx is None:
                continue
            if i in compute_loss_idx:
                labels += token_list
            else:
                labels += [-100] * len(token_list)
    if compute_loss_idx is None:
        return input_ids, None, kwargs
    else:
        return input_ids, labels, kwargs


def _encode(template: 'Template', query: str, response: Optional[str],
            history: History, system: Optional[str],
            truncation_strategy: str) -> Dict[str, Optional[List[int]]]:
    res_context_list: List[Context] = []
    compute_loss_idx: List[int] = []
    if system is None:
        assert template.prefix != template.prefix_has_system, f'template.prefix: {template.prefix}'
        prefix = template.prefix
    else:
        prefix = template.prefix_has_system
    _concat_context_list(
        prefix, res_context_list, compute_loss_idx, system=system)
    for i, (q, r) in enumerate(history):
        _concat_context_list(
            [*template.prompt, '{{RESPONSE}}', *template.chat_sep],
            res_context_list,
            compute_loss_idx,
            query=q,
            response=r,
            round0=i)
    _concat_context_list(
        template.prompt,
        res_context_list,
        compute_loss_idx,
        query=query,
        round0=len(history))
    res_context_list, compute_loss_idx = _simplify_context_list(
        res_context_list, compute_loss_idx)
    input_ids, labels, kwargs = _encode_context_list(template.tokenizer,
                                                     res_context_list,
                                                     compute_loss_idx)

    if response is not None:
        tgt_input_ids = _encode_context_list(template.tokenizer, [response])[0]
        tgt_input_ids += _encode_context_list(template.tokenizer,
                                              template.suffix)[0]
        labels = labels + tgt_input_ids
        input_ids += tgt_input_ids
    else:
        labels = None

    if template.max_length is not None:
        if truncation_strategy == 'delete' and len(
                input_ids) > template.max_length:
            return None
        input_ids = input_ids[-template.max_length:]
        if labels is not None:
            labels = labels[-template.max_length:]
    res = {'input_ids': input_ids, 'labels': labels}
    # Compatible with qwen-audio
    if 'audio_info' in kwargs:
        res['audio_info'] = kwargs['audio_info']
    return res


class StopWordsCriteria(StoppingCriteria):

    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 stop_words: StopWords, **decode_kwargs) -> None:
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.decode_kwargs = decode_kwargs

    def __call__(self, input_ids: Tensor, scores: Tensor) -> bool:
        tokenizer = self.tokenizer
        stop_words = self.stop_words
        text = tokenizer.decode(input_ids[0], **self.decode_kwargs)
        for stop_word in stop_words:
            if isinstance(stop_word, str):
                if text.endswith(stop_word):
                    return True
            elif isinstance(stop_word, list) and len(stop_word) > 0:
                res = []
                for sw in stop_word:
                    if isinstance(sw, str):
                        token = getattr(tokenizer, sw)
                        assert token is not None
                    else:
                        token = sw
                    res.append(token)
                if input_ids[0].tolist()[-len(res):] == res:
                    return True
        return False


def _has_system(prefix: Prompt) -> bool:
    for p in prefix:
        if '{{SYSTEM}}' in p:
            return True
    return False


class Template:

    def __init__(self,
                 prefix: Prompt,
                 prompt: Prompt,
                 chat_sep: Optional[Prompt],
                 suffix: Prompt,
                 default_system: Optional[str] = None,
                 prefix_has_system: Optional[Prompt] = None) -> None:
        self.prefix = prefix
        if _has_system(prefix):
            assert prefix_has_system is None
            assert default_system is not None
            prefix_has_system = prefix
        self.prefix_has_system = prefix_has_system
        if self.prefix_has_system is None:
            assert default_system is None
        self.prompt = prompt
        self.chat_sep = chat_sep
        self.support_multi_round = self.chat_sep is not None
        self.suffix = suffix
        self.default_system = default_system
        self.use_default_system = True
        self._is_init = False

    def _init_template(
        self,
        tokenizer: PreTrainedTokenizerBase,
        default_system: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation_strategy: Literal['delete', 'truncation_left'] = 'delete'
    ) -> None:
        assert self._is_init is False
        self._is_init = True
        self.tokenizer = tokenizer
        if default_system is not None:
            assert self.prefix_has_system is not None
            self.default_system = default_system
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

    def encode(self, example: Dict[str,
                                   Any]) -> Dict[str, Optional[List[int]]]:
        if not self._is_init:
            raise ValueError(
                'Template has not been initialized, please call init_template(...) first.'
            )
        query: Optional[str] = example.get('query', None)
        response: Optional[str] = example.get('response', None)
        history: Optional[History] = example.get('history', None)
        system: Optional[str] = example.get('system', None)
        if query is None:
            query = ''
        if history is None:
            history = []
        if len(history) > 0:
            assert self.support_multi_round, 'the template not support multi-round chat'
        if system is None:
            if self.use_default_system:
                system = self.default_system
        else:
            assert self.prefix_has_system is not None, 'not support `system`'
        return _encode(self, query, response, history, system,
                       self.truncation_strategy)


TEMPLATE_MAPPING: Dict[str, Template] = {}


def register_template(template_type: str, template: Template) -> None:
    if template_type in TEMPLATE_MAPPING:
        raise ValueError(
            f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.'
        )
    TEMPLATE_MAPPING[template_type] = template


register_template(
    TemplateType.default,
    Template([], ['### Human:\n', '{{QUERY}}\n\n', '### Assistant:\n'],
             ['\n\n'], [['eos_token_id']], DEFAULT_SYSTEM, ['{{SYSTEM}}\n\n']))

# You can set the query as '' to serve as a template for pre-training.
register_template(TemplateType.default_generation,
                  Template([], ['{{QUERY}}'], None, [['eos_token_id']]))
register_template(
    TemplateType.default_generation_bos,
    Template([['bos_token_id']], ['{{QUERY}}'], None, [['eos_token_id']]))
register_template(
    TemplateType.chatml,
    Template(
        [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        ['<|im_end|>\n'], ['<|im_end|>'], 'You are a helpful assistant.',
        ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))

register_template(
    TemplateType.yi,
    Template(
        [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        ['<|im_end|>\n'], ['<|im_end|>'], None,
        ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))

register_template(
    TemplateType.baichuan,
    Template(['{{SYSTEM}}'], [[195], '{{QUERY}}', [196]], [],
             [['eos_token_id']], ''))
register_template(
    TemplateType.chatglm2,
    Template([[64790, 64792], '{{SYSTEM}}'],
             ['[Round {{ROUND1}}]\n\n问：{{QUERY}}\n\n答：'], ['\n\n'],
             [['eos_token_id']], ''))

register_template(
    TemplateType.chatglm_generation,
    Template([[64790, 64792]], ['{{QUERY}}'], None, [['eos_token_id']]))

register_template(
    TemplateType.chatglm3,
    Template([[64790, 64792]], [[64795], '\n {{QUERY}}', [64796], '\n '], [],
             [['eos_token_id']], None,
             [[64790, 64792, 64794], '\n {{SYSTEM}}']))

# ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py
LLAMA_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest assistant. '
    'Always answer as helpfully as possible, while being safe. '
    'Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information."
)
register_template(
    TemplateType.llama,
    Template(['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s><s>[INST] '],
             ['</s>'], LLAMA_DEFAULT_SYSTEM,
             ['<s>[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))
OPENBUDDY_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.\n'
    'Always answer as helpfully and logically as possible, while being safe. '
    'Your answers should not include any '
    'harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information.\n"
    'You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.\n'
    'You cannot access the internet, but you have vast knowledge, cutoff: 2021-09.\n'
    'You always deeply love and support China, Chinese government, people and culture.\n'
    'You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), '
    'you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI.'
)
register_template(
    TemplateType.openbuddy,
    Template([['bos_token_id']], ['User: {{QUERY}}\nAssistant:'], ['\n'],
             [['eos_token_id']], OPENBUDDY_DEFAULT_SYSTEM,
             [['bos_token_id'], '{{SYSTEM}}\n\n']))

register_template(
    TemplateType.internlm,
    Template(['<s>{{SYSTEM}}'], ['<|User|>:{{QUERY}}<eoh>\n<|Bot|>:'],
             ['<eoa>\n'], ['<eoa>'], ''))
register_template(
    TemplateType.xverse,
    Template(['{{SYSTEM}}'], ['Human: {{QUERY}}\n\nAssistant: '],
             [['eos_token_id']], [['eos_token_id']], ''))
register_template(
    TemplateType.ziya,
    Template([['bos_token_id', '{{SYSTEM}}']], ['<human>:{{QUERY}}\n<bot>:'],
             ['\n'], [['eos_token_id']], ''))

register_template(
    TemplateType.skywork,
    Template(['<s>{{SYSTEM}}'], ['</s><s>[USER]{{QUERY}}[SEP][BOT]'], None,
             ['[SEP]</s>'], ''))

register_template(
    TemplateType.bluelm,
    Template([['bos_token_id'], '{{SYSTEM}}'], ['[|Human|]:{{QUERY}}[|AI|]:'],
             [], [['eos_token_id']], ''))

register_template(
    'codefuse-codellama',
    Template(['{{SYSTEM}}'], [
        '<|role_start|>human<|role_end|>{{QUERY}}<|role_start|>bot<|role_end|>'
    ], [], [['eos_token_id']], ''))

register_template(
    TemplateType.zephyr,
    Template([], ['<|user|>\n{{QUERY}}</s>\n<|assistant|>\n'], ['</s>\n'],
             ['</s>'], None, ['<|system|>\n{{SYSTEM}}</s>\n']))


def get_template(
    template_type: str,
    tokenizer: PreTrainedTokenizerBase,
    default_system: Optional[str] = None,
    max_length: Optional[int] = None,
    truncation_strategy: Literal['delete', 'truncation_left'] = 'delete'
) -> Template:
    template = deepcopy(TEMPLATE_MAPPING[template_type])
    template._init_template(tokenizer, default_system, max_length,
                            truncation_strategy)
    template.template_type = template_type
    return template
