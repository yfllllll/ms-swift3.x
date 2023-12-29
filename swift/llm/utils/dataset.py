# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import os
import re
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import json
import numpy as np
import pandas as pd
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets
from modelscope import MsDataset
from numpy.random import RandomState
from pandas import DataFrame
from tqdm.auto import tqdm

from swift.utils import (get_logger, get_seed, read_from_jsonl,
                         transform_jsonl_to_df)
from .preprocess import (AlpacaPreprocessor, ClsPreprocessor,
                         ComposePreprocessor, ConversationsPreprocessor,
                         PreprocessFunc, RenameColumnsPreprocessor,
                         SmartPreprocessor, TextGenerationPreprocessor)
from .template import History
from .utils import download_dataset


def _remove_useless_columns(dataset: HfDataset) -> HfDataset:
    k_list = []
    for k in dataset.features.keys():
        if k in {
                'query', 'response', 'rejected_response', 'system', 'history',
                'image'
        }:
            k_list.append(k)
    dataset = dataset.select_columns(k_list)
    return dataset


GetDatasetFunction = Callable[[], Union[HfDataset, Tuple[HfDataset,
                                                         Optional[HfDataset]]]]
SubsetSplit = Union[str, Tuple[str, str], List[str]]
DATASET_MAPPING: Dict[str, Dict[str, Any]] = {}

logger = get_logger()


class DatasetName:
    # general
    alpaca_en = 'alpaca-en'
    alpaca_zh = 'alpaca-zh'
    multi_alpaca_all = 'multi-alpaca-all'
    instinwild_en = 'instinwild-en'
    instinwild_zh = 'instinwild-zh'
    cot_en = 'cot-en'
    cot_zh = 'cot-zh'
    firefly_all_zh = 'firefly-all-zh'
    instruct_en = 'instruct-en'
    gpt4all_en = 'gpt4all-en'
    sharegpt_en = 'sharegpt-en'
    sharegpt_zh = 'sharegpt-zh'
    tutu_v2_sft_mixture = 'tutu-v2-sft-mixture'
    wikipedia_zh = 'wikipedia-zh'
    open_orca = 'open-orca'
    open_orca_gpt4 = 'open-orca-gpt4'
    sharegpt_gpt4 = 'sharegpt-gpt4'
    # agent
    damo_agent_zh = 'damo-agent-zh'
    damo_agent_mini_zh = 'damo-agent-mini-zh'
    agent_instruct_all_en = 'agent-instruct-all-en'
    # coding
    code_alpaca_en = 'code-alpaca-en'
    leetcode_python_en = 'leetcode-python-en'
    codefuse_python_en = 'codefuse-python-en'
    codefuse_evol_instruction_zh = 'codefuse-evol-instruction-zh'
    # medical
    medical_en = 'medical-en'
    medical_zh = 'medical-zh'
    medical_mini_zh = 'medical-mini-zh'
    # law
    lawyer_llama_zh = 'lawyer-llama-zh'
    tigerbot_law_zh = 'tigerbot-law-zh'
    # math
    blossom_math_zh = 'blossom-math-zh'
    school_math_zh = 'school-math-zh'
    open_platypus_en = 'open-platypus-en'
    # sql
    text2sql_en = 'text2sql-en'
    sql_create_context_en = 'sql-create-context-en'
    # text-generation
    advertise_gen_zh = 'advertise-gen-zh'
    dureader_robust_zh = 'dureader-robust-zh'
    # classification
    cmnli_zh = 'cmnli-zh'
    cmnli_mini_zh = 'cmnli-mini-zh'
    jd_sentiment_zh = 'jd-sentiment-zh'
    hc3_zh = 'hc3-zh'
    hc3_en = 'hc3-en'
    # other
    finance_en = 'finance-en'
    poetry_zh = 'poetry-zh'
    webnovel_zh = 'webnovel-zh'
    generated_chat_zh = 'generated-chat-zh'
    # example dataset for specific model
    cls_fudan_news_zh = 'cls-fudan-news-zh'  # seqgpt-560m
    ner_java_zh = 'ner-jave-zh'  # seqgpt-560m

    # multi-modal
    # vision
    coco_en = 'coco-en'
    coco_mini_en = 'coco-mini-en'
    capcha_images = 'capcha-images'
    # audio
    aishell1_zh = 'aishell1-zh'
    aishell1_mini_zh = 'aishell1-mini-zh'

    stack_exchange_paired = 'stack-exchange-paired'
    # dpo/hfrl dataset
    hh_rlhf = 'hh-rlhf'

    @classmethod
    def get_dataset_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__') or k == 'get_dataset_name_list':
                continue
            res.append(cls.__dict__[k])
        return res


def register_dataset(
        dataset_name: str,
        dataset_id_or_path: str,
        train_subset_split_list: Optional[List[SubsetSplit]] = None,
        val_subset_split_list: Optional[List[SubsetSplit]] = None,
        preprocess_func: Optional[PreprocessFunc] = None,
        get_function: Optional[GetDatasetFunction] = None,
        *,
        function_kwargs: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
        **kwargs
) -> Optional[Callable[[GetDatasetFunction], GetDatasetFunction]]:
    if preprocess_func is None:
        preprocess_func = SmartPreprocessor()
    if not exists_ok and dataset_name in DATASET_MAPPING:
        raise ValueError(
            f'The `{dataset_name}` has already been registered in the DATASET_MAPPING.'
        )
    if train_subset_split_list is None:
        train_subset_split_list = []
    if val_subset_split_list is None:
        val_subset_split_list = []
    if function_kwargs is None:
        function_kwargs = {}

    dataset_info = {
        'dataset_id_or_path': dataset_id_or_path,
        'train_subset_split_list': train_subset_split_list,
        'val_subset_split_list': val_subset_split_list,
        'preprocess_func': preprocess_func,
        **kwargs
    }
    if get_function is not None:
        if len(function_kwargs) > 0:
            get_function = partial(get_function, **function_kwargs)
        dataset_info['get_function'] = get_function
        DATASET_MAPPING[dataset_name] = dataset_info
        return

    def _register_dataset(
            get_function: GetDatasetFunction) -> GetDatasetFunction:
        _old_get_function = get_function
        if len(function_kwargs) > 0:
            get_function = partial(get_function, **function_kwargs)
        dataset_info['get_function'] = get_function
        DATASET_MAPPING[dataset_name] = dataset_info
        return _old_get_function

    return _register_dataset


def load_ms_dataset(
        dataset_id: str,
        subset_split_list: Optional[List[SubsetSplit]]) -> Optional[HfDataset]:
    if subset_split_list is None or len(subset_split_list) == 0:
        return None
    dataset_list = []
    for subset_split in subset_split_list:
        if isinstance(subset_split, str):
            subset_split = ('default', subset_split)
        assert len(subset_split) == 2
        subset_name, split = subset_split
        dataset = MsDataset.load(
            dataset_id, subset_name=subset_name, split=split).to_hf_dataset()
        dataset_list.append(dataset)
    return concatenate_datasets(dataset_list)


@register_dataset(
    DatasetName.text2sql_en,
    'AI-ModelScope/texttosqlv2_25000_v2', ['train'],
    tags=['chat', 'sql'])
@register_dataset(
    DatasetName.school_math_zh,
    'AI-ModelScope/school_math_0.25M', ['train'],
    tags=['chat', 'math'])
@register_dataset(
    DatasetName.gpt4all_en,
    'wyj123456/GPT4all', ['train'],
    tags=['chat', 'general'])
@register_dataset(
    DatasetName.cot_zh, 'YorickHe/CoT_zh', ['train'], tags=['chat', 'general'])
@register_dataset(
    DatasetName.cot_en, 'YorickHe/CoT', ['train'], tags=['chat', 'general'])
@register_dataset(
    DatasetName.instinwild_en,
    'wyj123456/instinwild', [('subset', 'train')],
    tags=['chat', 'general'])
@register_dataset(
    DatasetName.instinwild_zh,
    'wyj123456/instinwild', ['train'],
    tags=['chat', 'general'])
@register_dataset(
    DatasetName.code_alpaca_en,
    'wyj123456/code_alpaca_en', ['train'],
    tags=['chat', 'coding'])
@register_dataset(
    DatasetName.finance_en,
    'wyj123456/finance_en', ['train'],
    tags=['chat', 'financial'])
@register_dataset(
    DatasetName.alpaca_en,
    'AI-ModelScope/alpaca-gpt4-data-en', ['train'],
    tags=['chat', 'general', '🔥'])
def get_dataset_from_repo(
        dataset_id: str,
        train_subset_split_list: List[SubsetSplit],
        val_subset_split_list: Optional[List[SubsetSplit]],
        preprocess_func: PreprocessFunc,
        remove_useless_columns: bool = True,
        train_dataset_sample: int = -1,
        val_dataset_sample: int = -1) -> Tuple[HfDataset, Optional[HfDataset]]:
    dataset_list = []
    _iter = zip([train_subset_split_list, val_subset_split_list],
                [train_dataset_sample, val_dataset_sample])
    for subset_split_list, dataset_sample in _iter:
        dataset = load_ms_dataset(dataset_id, subset_split_list)
        if dataset is not None:
            if dataset_sample > 0 and len(dataset) > dataset_sample:
                random_state = np.random.RandomState(42)
                idxs = random_state.permutation(dataset_sample)
                dataset = dataset.select(idxs)
            dataset = preprocess_func(dataset)
            if remove_useless_columns:
                dataset = _remove_useless_columns(dataset)
        dataset_list.append(dataset)
    return tuple(dataset_list)


_multi_alpaca_subset_list = [
    'ar', 'de', 'es', 'fr', 'id', 'ja', 'ko', 'pt', 'ru', 'th', 'vi'
]

register_dataset(
    DatasetName.multi_alpaca_all,
    'damo/nlp_polylm_multialpaca_sft',
    [(subset, 'train') for subset in _multi_alpaca_subset_list],
    None,
    None,
    get_dataset_from_repo,
    tags=['chat', 'general', 'multilingual'],
    help="""language_list
    Language-key	Language	# examples
    ar	Arabic	14,671
    de	German	9,515
    es	Spanish	9,958
    fr	France	11,332
    id	Indonesian	12,117
    ja	Japanese	10,191
    ko	Korean	14,402
    pt	Portuguese	10,825
    ru	Russian	14,286
    th	Thai	11,496
    vi	Vietnamese	13,908
""")


def _concat_inst_inp_alpaca_zh(inst: str, inp: str) -> str:
    if inp.startswith('输入：'):
        inp = inp[3:]
    return f'{inst}\n{inp}'


register_dataset(
    DatasetName.alpaca_zh,
    'AI-ModelScope/alpaca-gpt4-data-zh', ['train'],
    None,
    AlpacaPreprocessor(concat_inst_inp=_concat_inst_inp_alpaca_zh),
    get_dataset_from_repo,
    tags=['chat', 'general', '🔥'])


def _preprocess_vision_dataset(dataset: HfDataset) -> HfDataset:
    prompt = 'please describe the image'
    image_key = 'image'
    response_key = 'caption'

    dataset._info.features._column_requires_decoding['image'] = False
    query_format = f'Picture 1:<img>{{image_path}}</img>\n{prompt}'
    query = []
    response = []
    for d in tqdm(dataset):
        query.append(query_format.format(image_path=d[image_key]['path']))
        if '&&' in d[response_key]:
            d[response_key] = d[response_key].split('&&')[0]
        response.append(d[response_key])
    dataset = HfDataset.from_dict({'query': query, 'response': response})
    return dataset


register_dataset(
    DatasetName.coco_en,
    'modelscope/coco_2014_caption', [('coco_2014_caption', 'train')],
    [('coco_2014_caption', 'validation')],
    _preprocess_vision_dataset,
    get_dataset_from_repo,
    tags=['chat', 'multi-modal', 'vision'])

register_dataset(
    DatasetName.coco_mini_en,
    'modelscope/coco_2014_caption', [('coco_2014_caption', 'train')],
    [('coco_2014_caption', 'validation')],
    _preprocess_vision_dataset,
    get_dataset_from_repo,
    function_kwargs={
        'train_dataset_sample': 20000,
        'val_dataset_sample': 200
    },
    tags=['chat', 'multi-modal', 'vision', '🔥'])


def _preprocess_aishell1_dataset(dataset: HfDataset) -> HfDataset:
    prompt = '语音转文本'
    audio_key = 'Audio:FILE'
    response_key = 'Text:LABEL'
    query_format = f'Audio 1:<audio>{{audio_path}}</audio>\n{prompt}'
    query = []
    response = []
    for d in tqdm(dataset):
        query.append(query_format.format(audio_path=d[audio_key]))
        response.append(d[response_key].replace(' ', ''))
    dataset = HfDataset.from_dict({'query': query, 'response': response})
    return dataset


register_dataset(
    DatasetName.aishell1_zh,
    'speech_asr/speech_asr_aishell1_trainsets', ['train', 'validation'],
    ['test'],
    _preprocess_aishell1_dataset,
    get_dataset_from_repo,
    tags=['chat', 'multi-modal', 'audio'])

register_dataset(
    DatasetName.aishell1_mini_zh,
    'speech_asr/speech_asr_aishell1_trainsets', ['validation'], ['test'],
    _preprocess_aishell1_dataset,
    get_dataset_from_repo,
    function_kwargs={'val_dataset_sample': 200},
    tags=['chat', 'multi-modal', 'audio', '🔥'])


def _repair_agent_conversations(conversations: str,
                                use_mini: bool) -> Dict[str, str]:
    if use_mini:
        pattern = r'\d\. {"plugin_name": "(.+?)"'
    else:
        pattern = r'\d\. {"(?:plugin_)?name": "(.+?)"'

    idx = conversations.find(r"'from': 'user")
    if idx == -1:
        return
    # remove dirty data
    find_list = re.findall(pattern, conversations[:idx])
    if len(set(find_list)) <= 1:
        return
    conversations = ast.literal_eval(conversations)
    if len(conversations) == 1:
        return
    return conversations


register_dataset(
    DatasetName.damo_agent_mini_zh,
    'damo/MSAgent-Bench', ['train'], ['validation'],
    ConversationsPreprocessor(
        repair_conversations=partial(
            _repair_agent_conversations, use_mini=True)),
    get_dataset_from_repo,
    tags=['chat', 'agent', 'multi-round', '🔥'])
register_dataset(
    DatasetName.damo_agent_zh,
    'damo/MSAgent-Bench', ['train'], ['validation'],
    ConversationsPreprocessor(
        repair_conversations=partial(
            _repair_agent_conversations, use_mini=False)),
    get_dataset_from_repo,
    tags=['chat', 'agent', 'multi-round'])

advertise_gen_prompt = """Task: Generating advertisements based on keywords.
Keywords: {query}
Advertisements: """
register_dataset(
    DatasetName.advertise_gen_zh,
    'lvjianjin/AdvertiseGen', ['train'], ['validation'],
    TextGenerationPreprocessor(advertise_gen_prompt, 'content', 'summary'),
    get_dataset_from_repo,
    tags=['text-generation', '🔥'])

_firefly_kind_list = [
    'ProseGeneration', 'MRC', 'JinYongGeneration', 'TextCorrection',
    'ClassicalChinese', 'BELLE', 'StoryGeneration', 'Couplet', 'Cot',
    'Dictionary', 'Translation', 'Program', 'SentimentAnalyze', 'OpenQA',
    'AncientPoem', 'TextMatching', 'NLI', 'Summary', 'KeywordRecognition',
    'ProductDesc', 'LyricGeneration', 'Composition', 'MusicComment', 'NER'
]


def _preprocess_firefly(dataset: List[Dict[str, str]],
                        kind_list: List[str]) -> HfDataset:
    kind_set = set(kind_list)
    query: List[str] = []
    response: List[str] = []
    for d in tqdm(dataset):
        if d['kind'] not in kind_set:
            continue
        query.append(d['input'])
        response.append(d['target'])

    return HfDataset.from_dict({
        'query': query,
        'response': response,
    })


@register_dataset(
    DatasetName.firefly_all_zh,
    'wyj123456/firefly',
    preprocess_func=_preprocess_firefly,
    tags=['chat', 'general'],
    function_kwargs={'kind_list': _firefly_kind_list})
def get_firefly_zh_dataset(dataset_id: str, preprocess_func,
                           kind_list: List[str], **kwargs) -> HfDataset:
    file = 'firefly-train-1.1M.jsonl'
    dataset_dir = download_dataset(dataset_id, [file])
    fpath = os.path.join(dataset_dir, file)
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.replace('}{', '},{')
        text = f'[{text}]'
        dataset = json.loads(text)
    return preprocess_func(dataset, kind_list)


register_dataset(
    DatasetName.poetry_zh,
    'modelscope/chinese-poetry-collection', ['train'], ['test'],
    RenameColumnsPreprocessor({'text1': 'response'}),
    get_dataset_from_repo,
    tags=['text-generation', 'poetry'])

register_dataset(
    DatasetName.instruct_en,
    'wyj123456/instruct', ['train'],
    None,
    RenameColumnsPreprocessor({
        'prompt': 'query',
        'completion': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'general'])

register_dataset(
    DatasetName.cmnli_zh,
    'clue', [('cmnli', 'train')], [('cmnli', 'validation')],
    ClsPreprocessor(['neutral', 'entailment', 'contradiction'],
                    'Natural Language Inference', True),
    get_dataset_from_repo,
    tags=['text-generation', 'classification'])

register_dataset(
    DatasetName.cmnli_mini_zh,
    'clue', [('cmnli', 'train')], [('cmnli', 'validation')],
    ClsPreprocessor(['neutral', 'entailment', 'contradiction'],
                    'Natural Language Inference', True),
    get_dataset_from_repo,
    function_kwargs={
        'train_dataset_sample': 20000,
        'val_dataset_sample': 200
    },
    tags=['text-generation', 'classification', '🔥'])

register_dataset(
    DatasetName.jd_sentiment_zh,
    'DAMO_NLP/jd', ['train'], ['validation'],
    ClsPreprocessor(['negative', 'positive'], 'Sentiment Classification',
                    False),
    get_dataset_from_repo,
    tags=['text-generation', 'classification', '🔥'])


def _preprocess_dureader_robust(dataset: HfDataset) -> HfDataset:
    prompt = """Task: Question Generation
Context: {context}
Answer: {answer}
Question: """
    query = []
    response = []
    for d in dataset:
        answer, context = d['text1'].split('[SEP]')
        q = prompt.format(context=context, answer=answer)
        query.append(q)
        response.append(d['text2'])
    return HfDataset.from_dict({'query': query, 'response': response})


register_dataset(
    DatasetName.dureader_robust_zh,
    'modelscope/DuReader_robust-QG', ['train', 'validation'], ['test'],
    _preprocess_dureader_robust,
    get_dataset_from_repo,
    tags=['text-generation', '🔥'])

register_dataset(
    DatasetName.medical_en,
    'huangjintao/medical_zh', [('en', 'train'), ('en', 'val')],
    [('en', 'test')],
    RenameColumnsPreprocessor({
        'input': 'query',
        'output': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'medical'])

register_dataset(
    DatasetName.stack_exchange_paired,
    'AI-ModelScope/stack-exchange-paired', [('default', 'train')],
    None,
    RenameColumnsPreprocessor({
        'question': 'query',
        'response_j': 'response',
        'response_k': 'rejected_response',
    }),
    get_dataset_from_repo,
    tags=['hfrl', 'dpo', 'pairwise'])


def process_hh_rlhf(dataset):

    def extract_anthropic_prompt(prompt_and_response):
        """Extract the anthropic prompt from a prompt and response pair."""
        search_term = '\n\nAssistant:'
        search_term_idx = prompt_and_response.rfind(search_term)
        assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
        return prompt_and_response[:search_term_idx + len(search_term)]

    def reorganize_row_simple(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample['chosen'])
        return {
            'query': prompt,
            'response': sample['chosen'][len(prompt):],
            'rejected_response': sample['rejected'][len(prompt):],
        }

    return dataset.map(reorganize_row_simple)


register_dataset(
    DatasetName.hh_rlhf,
    'AI-ModelScope/hh-rlhf', [('harmless-base', 'train')],
    [('harmless-base', 'test')],
    process_hh_rlhf,
    get_dataset_from_repo,
    tags=['hfrl', 'dpo', 'pairwise'])

register_dataset(
    DatasetName.medical_zh,
    'huangjintao/medical_zh', [('zh', 'train'), ('zh', 'val')],
    [('zh', 'test')],
    RenameColumnsPreprocessor({
        'instruction': 'query',
        'output': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'medical'])

register_dataset(
    DatasetName.medical_mini_zh,
    'huangjintao/medical_zh', [('zh', 'train'), ('zh', 'val')],
    [('zh', 'test')],
    RenameColumnsPreprocessor({
        'instruction': 'query',
        'output': 'response'
    }),
    get_dataset_from_repo,
    function_kwargs={'train_dataset_sample': 50000},
    tags=['chat', 'medical'])


def _preprocess_sharegpt(dataset: HfDataset) -> HfDataset:
    query = []
    response = []
    history: List[History] = []
    for d in tqdm(dataset):
        conversation = ast.literal_eval(d['conversation'])
        query.append(conversation[-1]['human'])
        response.append(conversation[-1]['assistant'])
        h = []
        for c in conversation[:-1]:
            h.append([c['human'], c['assistant']])
        history.append(h)
    return HfDataset.from_dict({
        'query': query,
        'response': response,
        'history': history
    })


_sharegpt_zh_subset_list = ['common-zh', 'computer-zh', 'unknow-zh']

_sharegpt_en_subset_list = ['common-en', 'computer-en']

register_dataset(
    DatasetName.sharegpt_zh,
    'huangjintao/sharegpt',
    [(subset, 'train') for subset in _sharegpt_zh_subset_list],
    None,
    _preprocess_sharegpt,
    get_dataset_from_repo,
    tags=['chat', 'general', 'multi-round'])

register_dataset(
    DatasetName.sharegpt_en,
    'huangjintao/sharegpt',
    [(subset, 'train') for subset in _sharegpt_en_subset_list],
    None,
    _preprocess_sharegpt,
    get_dataset_from_repo,
    tags=['chat', 'general', 'multi-round'])


def _preprocess_capcha_images(dataset: HfDataset) -> HfDataset:
    dataset = dataset.rename_columns({
        'solution': 'response',
    })

    def add_system(row):
        row['query'] = 'CAPTCHA:'
        return row

    dataset = dataset.map(add_system)
    return dataset


register_dataset(
    DatasetName.capcha_images,
    'AI-ModelScope/captcha-images', [('default', 'train')],
    [('default', 'validation')],
    _preprocess_capcha_images,
    get_dataset_from_repo,
    tags=['chat', 'multi-modal', 'vision', '🔥'])

register_dataset(
    DatasetName.cls_fudan_news_zh,
    'damo/zh_cls_fudan-news', ['train'],
    None,
    RenameColumnsPreprocessor({
        'prompt': 'query',
        'answer': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'classification'])

register_dataset(
    DatasetName.ner_java_zh,
    'damo/zh_ner-JAVE', ['train'],
    None,
    RenameColumnsPreprocessor({
        'prompt': 'query',
        'answer': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'ner'])

register_dataset(
    DatasetName.codefuse_python_en,
    'codefuse-ai/CodeExercise-Python-27k', ['train'],
    None,
    ConversationsPreprocessor(
        'human',
        'bot',
        conversations_key='chat_rounds',
        from_key='role',
        value_key='content'),
    get_dataset_from_repo,
    tags=['chat', 'coding', '🔥'])


def _preprocess_blossom_math(dataset: HfDataset) -> HfDataset:
    response = []
    for d in tqdm(dataset):
        output, answer = d['output'], d['answer']
        response.append(f'{output}\n\nAnswer: {answer}')
    return HfDataset.from_dict({
        'query': dataset['input'],
        'response': response
    })


register_dataset(
    DatasetName.blossom_math_zh,
    'AI-ModelScope/blossom-math-v2', ['train'],
    None,
    _preprocess_blossom_math,
    get_dataset_from_repo,
    tags=['chat', 'math', '🔥'])

register_dataset(
    DatasetName.sql_create_context_en,
    'AI-ModelScope/sql-create-context', ['train'],
    None,
    ComposePreprocessor([
        RenameColumnsPreprocessor({
            'question': 'instruction',
            'context': 'input',
            'answer': 'output'
        }),
        AlpacaPreprocessor(),
    ]),
    get_dataset_from_repo,
    tags=['chat', 'sql', '🔥'])

register_dataset(
    DatasetName.lawyer_llama_zh,
    'AI-ModelScope/lawyer_llama_data', ['train'],
    None,
    RenameColumnsPreprocessor({
        'instruction': 'query',
        'output': 'response',
        'history': '_'
    }),
    get_dataset_from_repo,
    tags=['chat', 'law', '🔥'])


def _preprocess_tigerbot_law(dataset: HfDataset) -> HfDataset:
    prompt = """{type}
{title}
"""
    response = []
    for d in tqdm(dataset):
        cur_prompt = prompt.format(type=d['type'], title=d['title'])
        for i in range(1, 4):
            chapter = d[f'chapter{i}']
            if chapter is not None:
                cur_prompt += f'{chapter}'
        cur_prompt += f'{d["content"]}'
        response.append(cur_prompt)
    return HfDataset.from_dict({
        'response': response,
    })


register_dataset(
    DatasetName.tigerbot_law_zh,
    'AI-ModelScope/tigerbot-law-plugin', ['train'],
    None,
    _preprocess_tigerbot_law,
    get_dataset_from_repo,
    tags=['text-generation', 'law', 'pretrained', '🔥'])


def _preprocess_leetcode_python(dataset: HfDataset) -> HfDataset:
    query = []
    response = []
    for d in dataset:
        code_with_problem = d['code_with_problem']
        idx = code_with_problem.find('```python')
        idx2 = code_with_problem.rfind('```python')
        assert idx == idx2
        problem = code_with_problem[:idx]
        if problem.startswith('# '):
            problem = problem[2:]
        code = code_with_problem[idx:].strip()
        explanation = d['explanation_only']
        query.append(problem)
        response.append(f'{code}\n\n{explanation}')
    return HfDataset.from_dict({'query': query, 'response': response})


register_dataset(
    DatasetName.leetcode_python_en,
    'AI-ModelScope/leetcode-solutions-python', ['train'],
    None,
    _preprocess_leetcode_python,
    get_dataset_from_repo,
    tags=['chat', 'coding', '🔥'])

_agent_instruct_subset_list = [
    'alfworld', 'db', 'kg', 'mind2web', 'os', 'webshop'
]


def _repair_conversations_agent_instruct(s: str) -> str:
    s = s.replace('}\n {', '},\n {')
    return ast.literal_eval(s)


register_dataset(
    DatasetName.agent_instruct_all_en,
    'huangjintao/AgentInstruct_copy',
    [(subset, 'train') for subset in _agent_instruct_subset_list],
    None,
    ConversationsPreprocessor(
        'human',
        'gpt',
        repair_conversations=_repair_conversations_agent_instruct),
    get_dataset_from_repo,
    tags=['chat', 'agent', 'multi-round', '🔥'])

register_dataset(
    DatasetName.codefuse_evol_instruction_zh,
    'codefuse-ai/Evol-instruction-66k', ['train'],
    None,
    RenameColumnsPreprocessor({
        'instruction': 'query',
        'output': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'coding', '🔥'])

hc3_chinese_subset = [
    'baike', 'open_qa', 'nlpcc_dbqa', 'finance', 'medicine', 'law',
    'psychology'
]


def _preprocess_hc3(dataset: HfDataset) -> HfDataset:
    prompt = """Classification Task: Are the following responses from a human or from ChatGPT?
Question: {question}
Answer: {answer}
Category: Human, ChatGPT
Output: """
    query = []
    response = []
    for d in dataset:
        question = d['question']
        for h in d['human_answers']:
            query.append(prompt.format(question=question, answer=h))
            response.append('Human')
        for c in d['chatgpt_answers']:
            query.append(prompt.format(question=question, answer=c))
            response.append('ChatGPT')
    return HfDataset.from_dict({'query': query, 'response': response})


register_dataset(
    DatasetName.hc3_zh,
    'simpleai/HC3-Chinese',
    [[subset, 'train'] for subset in hc3_chinese_subset], [],
    _preprocess_hc3,
    get_dataset_from_repo,
    tags=['text-generation', 'classification', '🔥'])

register_dataset(
    DatasetName.hc3_en,
    'simpleai/HC3', [[subset, 'train'] for subset in ['finance', 'medicine']],
    [],
    _preprocess_hc3,
    get_dataset_from_repo,
    tags=['text-generation', 'classification', '🔥'])

register_dataset(
    DatasetName.tutu_v2_sft_mixture,
    'AI-ModelScope/tulu-v2-sft-mixture', ['train'], [],
    None,
    get_dataset_from_repo,
    tag=['chat', 'multilingual', 'general', 'multi-round'])
register_dataset(
    DatasetName.webnovel_zh,
    'AI-ModelScope/webnovel_cn', ['train'], [],
    None,
    get_dataset_from_repo,
    tags=['chat', 'novel'])
register_dataset(
    DatasetName.generated_chat_zh,
    'AI-ModelScope/generated_chat_0.4M', ['train'], [],
    None,
    get_dataset_from_repo,
    tags=['chat', 'character-dialogue'])
register_dataset(
    DatasetName.wikipedia_zh,
    'AI-ModelScope/wikipedia-cn-20230720-filtered', ['train'],
    None,
    RenameColumnsPreprocessor({'completion': 'response'}),
    get_dataset_from_repo,
    tags=['text-generation', 'general', 'pretrained'])
register_dataset(
    DatasetName.open_platypus_en,
    'AI-ModelScope/Open-Platypus', ['train'],
    None,
    None,
    get_dataset_from_repo,
    tags=['chat', 'math'])
register_dataset(
    DatasetName.open_orca_gpt4,
    'AI-ModelScope/OpenOrca', ['train'],
    None,
    RenameColumnsPreprocessor({'question': 'query'}),
    get_dataset_from_repo,
    tags=['chat', 'multilingual', 'general'])
register_dataset(
    DatasetName.open_orca,
    'AI-ModelScope/OpenOrca', [['3_5M', 'train']],
    None,
    RenameColumnsPreprocessor({'question': 'query'}),
    get_dataset_from_repo,
    tags=['chat', 'multilingual', 'general'])
register_dataset(
    DatasetName.sharegpt_gpt4,
    'AI-ModelScope/sharegpt_gpt4',
    [[subset, 'train']
     for subset in ['default', 'V3_format', 'zh_38K_format']],
    None,
    ConversationsPreprocessor('human', 'gpt', error_strategy='delete'),
    get_dataset_from_repo,
    tags=['chat', 'multilingual', 'general', 'multi-round'])


def add_self_cognition_dataset(
        train_dataset: HfDataset, dataset_sample: int,
        model_name: Tuple[str, Optional[str]],
        model_author: Tuple[str, Optional[str]]) -> None:
    assert model_name[0] is not None
    assert model_author[0] is not None
    if model_name[1] is None:
        model_name = (model_name[0], model_name[0])
    if model_author[1] is None:
        model_author = (model_author[0], model_author[0])
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'data',
        'self_cognition.jsonl')
    assert os.path.exists(dataset_path)
    dataset = load_dataset_from_local([dataset_path], SmartPreprocessor())
    response = []
    for d in dataset:
        if d['tag'] == 'zh':
            model_n, model_a = model_name[0], model_author[0]
        else:
            model_n, model_a = model_name[1], model_author[1]

        r = d['response'].replace('{{NAME}}',
                                  model_n).replace('{{AUTHOR}}', model_a)
        response.append(r)
    dataset = dataset.remove_columns('response').add_column(
        'response', response).remove_columns('tag')

    random_state = RandomState(42)
    idx = random_state.permutation(len(dataset))[:dataset_sample]
    dataset_sample -= len(idx)
    if dataset_sample > 0:
        idx2 = random_state.choice(len(dataset), dataset_sample)
        idx = np.concatenate([idx, idx2], axis=0)
    dataset = dataset.select(idx)
    if train_dataset is None:
        return dataset
    else:
        return concatenate_datasets([train_dataset, dataset])


def _check_dataset(
    dataset: Optional[None],
    check_dataset_strategy: Literal['none', 'discard', 'error', 'warning']
) -> HfDataset:
    if check_dataset_strategy == 'none' or dataset is None:
        return dataset
    idx_list = []
    if check_dataset_strategy == 'error':
        assert len(
            set(dataset.features.keys())
            - set(['query', 'response', 'system', 'history'])) == 0
    has_query = 'query' in dataset.features
    has_history = 'history' in dataset.features
    has_system = 'system' in dataset.features
    is_modified = False
    for i, d in enumerate(tqdm(dataset)):
        if not isinstance(d['response'], str):
            is_modified = True
            if check_dataset_strategy == 'discard':
                continue
            elif check_dataset_strategy == 'warning':
                logger.warning(f"d['response']: {d['response']}, i: {i}")
                continue
            else:
                raise ValueError(f"d['response']: {d['response']}, i: {i}")
        if has_query and not isinstance(d['response'], str):
            is_modified = True
            if check_dataset_strategy == 'discard':
                continue
            elif check_dataset_strategy == 'warning':
                logger.warning(f"d['query']: {d['query']}, i: {i}")
                continue
            else:
                raise ValueError(f"d['query']: {d['query']}, i: {i}")
        if has_history and not isinstance(d['history'], (list, type(None))):
            is_modified = True
            if check_dataset_strategy == 'discard':
                continue
            elif check_dataset_strategy == 'warning':
                logger.warning(f"d['history']: {d['history']}, i: {i}")
                continue
            else:
                raise ValueError(f"d['history']: {d['history']}, i: {i}")
        if has_system and not isinstance(d['system'], str):
            is_modified = True
            if check_dataset_strategy == 'discard':
                continue
            elif check_dataset_strategy == 'warning':
                logger.warning(f"d['system']: {d['system']}, i: {i}")
                continue
            else:
                raise ValueError(f"d['system']: {d['system']}, i: {i}")
        idx_list.append(i)
    if is_modified:
        dataset = dataset.select(idx_list)
    assert len(dataset) > 0
    return dataset


def get_dataset(
    dataset_name_list: Union[List[str], str],
    dataset_test_ratio: float = 0.,
    dataset_seed: Union[RandomState, int] = 42,
    check_dataset_strategy: Literal['none', 'discard', 'error',
                                    'warning'] = 'none'
) -> Tuple[HfDataset, Optional[HfDataset]]:
    """Returns train_dataset and val_dataset"""
    if isinstance(dataset_name_list, str):
        dataset_name_list = [dataset_name_list]
    train_dataset_list: List[HfDataset] = []
    val_dataset_list: List[HfDataset] = []
    random_state = dataset_seed
    if isinstance(dataset_seed, int):
        random_state = RandomState(dataset_seed)
    for dataset_name in dataset_name_list:
        dataset_info = DATASET_MAPPING[dataset_name]
        get_function: GetDatasetFunction = dataset_info['get_function']
        dataset = get_function(
            dataset_info['dataset_id_or_path'],
            train_subset_split_list=dataset_info['train_subset_split_list'],
            val_subset_split_list=dataset_info['val_subset_split_list'],
            preprocess_func=dataset_info['preprocess_func'])
        train_d: HfDataset
        if isinstance(dataset, (list, tuple)):
            train_d, val_d = dataset
        else:
            train_d, val_d = dataset, None
        assert train_d is not None or val_d is not None
        if val_d is None and dataset_test_ratio > 0:
            dataset_dict = train_d.train_test_split(
                dataset_test_ratio, seed=get_seed(random_state))
            train_d, val_d = dataset_dict['train'], dataset_dict['test']
        if train_d is not None:
            train_dataset_list.append(train_d)
        if val_d is not None:
            val_dataset_list.append(val_d)

    train_dataset = None
    if len(train_dataset_list) > 0:
        train_dataset = concatenate_datasets(train_dataset_list)
    val_dataset = None
    if len(val_dataset_list) > 0:
        val_dataset = concatenate_datasets(val_dataset_list)
    if check_dataset_strategy != 'none':
        logger.info('check dataset...')
        logger.info(f"check_dataset_strategy: '{check_dataset_strategy}'")
    train_dataset = _check_dataset(train_dataset, check_dataset_strategy)
    val_dataset = _check_dataset(val_dataset, check_dataset_strategy)
    return train_dataset, val_dataset


def load_dataset_from_local(
        dataset_path_list: Optional[Union[str, List[str]]],
        preprocess_func: PreprocessFunc) -> Optional[HfDataset]:
    if isinstance(dataset_path_list, str):
        dataset_path_list = [dataset_path_list]
    if dataset_path_list is None or len(dataset_path_list) == 0:
        return None
    assert isinstance(dataset_path_list, (list, tuple))

    dataset_list = []
    for dataset_path in dataset_path_list:
        assert isinstance(dataset_path, str)
        df: DataFrame
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.jsonl'):
            df = transform_jsonl_to_df(read_from_jsonl(dataset_path))
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                obj_list = json.load(f)
            df = transform_jsonl_to_df(obj_list)
        else:
            raise ValueError(
                'The custom dataset only supports CSV format or JSONL format. You can refer to the link '
                '`https://github.com/modelscope/swift/blob/main/docs/source/LLM/自定义与拓展.md#注册数据集的方式` '
                'for more information.')
        dataset = HfDataset.from_dict(df.to_dict(orient='list'))
        dataset_list.append(preprocess_func(dataset))
    return concatenate_datasets(dataset_list)


def get_custom_dataset(_: str, train_subset_split_list: Union[str, List[str]],
                       val_subset_split_list: Optional[Union[str, List[str]]],
                       preprocess_func: PreprocessFunc,
                       **kwargs) -> Tuple[HfDataset, Optional[HfDataset]]:
    train_dataset = load_dataset_from_local(train_subset_split_list,
                                            preprocess_func)
    val_dataset = load_dataset_from_local(val_subset_split_list,
                                          preprocess_func)
    return train_dataset, val_dataset
