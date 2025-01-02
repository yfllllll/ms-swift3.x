# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import os
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import Image
from datasets import IterableDataset as HfIterableDataset
from datasets import Value

from swift.llm import history_to_messages
from swift.utils import get_logger

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

standard_keys = ['messages', 'rejected_response', 'label', 'images', 'videos', 'audios', 'tools', 'objects', 'bbox', 'yolo', 'class_names']

logger = get_logger()


def get_features_dataset(dataset: DATASET_TYPE) -> DATASET_TYPE:
    if dataset.features is None:
        assert isinstance(dataset, HfIterableDataset)
        dataset = dataset._resolve_features()
    return dataset


class RowPreprocessor:

    def __init__(self,
                 *,
                 columns_mapping: Optional[Dict[str, str]] = None,
                 dataset_sample: Optional[int] = None,
                 random_state: Union[np.random.RandomState, int, None] = None,
                 traceback_limit: int = 10) -> None:
        self.columns_mapping = columns_mapping or {}
        images_keys = ['images', 'image']
        audios_keys = ['audios', 'audio']
        videos_keys = ['videos', 'video']
        for mm_type in ['images', 'audios', 'videos']:
            keys = locals()[f'{mm_type}_keys']
            for key in keys:
                self.columns_mapping[key] = mm_type

        self.traceback_limit = traceback_limit
        self._traceback_counter = 0
        self.dataset_sample = dataset_sample
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

    @staticmethod
    def _check_messages(row: Dict[str, Any]) -> None:
        if 'messages' not in row:
            return
        messages = row['messages']
        assert len(messages) > 0, f'messages: {messages}'
        if messages[0]['role'] == 'system':
            messages = messages[1:]
        if messages and messages[0]['role'] == 'assistant':
            messages = [{'role': 'user', 'content': ''}] + messages  # pretrain
        for user_message, assistant_message in zip(messages[::2], messages[1::2]):
            if (user_message['role'] not in {'user', 'tool'} or 'content' not in user_message
                    or user_message['content'] is None):
                raise ValueError(f'user_message: {user_message}')
            if (assistant_message['role'] not in {'assistant'} or 'content' not in assistant_message
                    or assistant_message['content'] in {'', None}):
                raise ValueError(f'assistant_message: {assistant_message}')

    @staticmethod
    def _cast_images(row: Dict[str, Any]) -> None:
        images = row.get('images')

        if isinstance(images, str) or isinstance(images, list) and images and isinstance(images[0], str):
            if isinstance(images, str):
                images = [images]
            for i, image in enumerate(images):
                images[i] = {'bytes': None, 'path': image}
            row['images'] = images
        elif isinstance(images, dict):
            row['images'] = [images]

    @staticmethod
    def _check_rejected_response(row: Dict[str, Any]) -> None:
        if 'rejected_messages' in row:
            chosen_messages = row['messages']
            rejected_messages = row['rejected_messages']
            messages = []
            rejected_response = None
            for chosen_user, chosen_assistant, rejected_user, rejected_assistant in zip(
                    chosen_messages[::2], chosen_messages[1::2], rejected_messages[::2], rejected_messages[1::2]):
                assert chosen_user == rejected_user
                messages.append(chosen_user)
                messages.append(chosen_assistant)
                if chosen_assistant != rejected_assistant:
                    rejected_response = rejected_assistant['content']
            row['messages'] = messages
            row['rejected_response'] = rejected_response

        if 'rejected_response' in row:
            messages = row['messages']
            rejected_response = row['rejected_response']
            if rejected_response is None or rejected_response == messages[-1]['content']:
                raise ValueError(f'rejected_response: {rejected_response}')

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def prepare_dataset(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        return dataset

    @staticmethod
    def batched_to_rows(batched_row: Dict[str, Any]):
        keys = list(batched_row.keys())
        batch_size = len(batched_row[keys[0]])
        return [{key: batched_row[key][i] for key in keys} for i in range(batch_size)]

    @staticmethod
    def rows_to_batched(rows: List[Dict[str, Any]]):
        batched = {}
        for i, row in enumerate(rows):
            for k, v in row.items():
                if k not in batched:
                    batched[k] = [None] * i
                batched[k].append(v)
        # Make all the lengths of v the same.
        batched = {k: v + [None] * (len(rows) - len(v)) for k, v in batched.items()}
        return batched

    @staticmethod
    def _fix_streaming_keys(row):
        for k in list(row.keys()):
            if k.startswith('__@'):
                new_k = k[len('__@'):]
                row[new_k] = row.pop(k)

    def batched_preprocess(self, batched_row: Dict[str, Any], *, strict: bool) -> Dict[str, Any]:
        batched_row = dict(batched_row)
        assert len(batched_row) > 0
        self._fix_streaming_keys(batched_row)
        rows = self.batched_to_rows(batched_row)

        new_rows = []
        for row in rows:
            try:
                row = self.preprocess(row)
                # support [row1, row2, ...]
                if row is None:
                    row = []
                if isinstance(row, dict):
                    row = [row]
                for r in row:
                    self._check_messages(r)
                    self._check_rejected_response(r)
                    self._cast_images(r)
            except Exception:
                if strict:
                    logger.warning('To avoid errors, you can pass `strict=False`.')
                    raise
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback
                    print(traceback.format_exc())
                    logger.error('👆👆👆There are errors in the dataset, the data will be deleted')
                    self._traceback_counter += 1
                row = []
            new_rows += row
        res = self.rows_to_batched(new_rows)

        if len(res) == 0:
            res['messages'] = []

        return res

    @staticmethod
    def safe_rename_columns(dataset: DATASET_TYPE, columns_mapping: Dict[str, Any]) -> DATASET_TYPE:
        dataset = get_features_dataset(dataset)
        columns_keys = {k.lower(): k for k in dataset.features.keys()}  # lower -> lower/upper
        safe_columns_mapping = {
            columns_keys[k.lower()]: v
            for k, v in columns_mapping.items() if k.lower() in columns_keys
        }

        counter = Counter(safe_columns_mapping.values())
        for k, new_k in list(safe_columns_mapping.items()):
            if counter[new_k] > 1:
                # For example, if "response" and "answer" match, then no processing is done.
                safe_columns_mapping.pop(k)
                continue

        # e.g. Keep {'query': 'query'} to ensure that the query has the highest priority.
        safe_columns_mapping = {k: v for k, v in safe_columns_mapping.items() if k != v}
        if safe_columns_mapping:
            dataset = dataset.rename_columns(safe_columns_mapping)

        if isinstance(dataset, HfIterableDataset):
            # fix: https://github.com/huggingface/datasets/issues/6408
            columns_mapping = {k: f'__@{k}' for k in standard_keys if k in dataset.features}
            if columns_mapping:
                dataset = dataset.rename_columns(columns_mapping)

        return dataset

    @staticmethod
    @contextmanager
    def _patch_arrow_writer():
        # fix AI-ModelScope/ms_agent_for_agentfabric:all
        from datasets.arrow_writer import ArrowWriter

        def _new_init(self, schema=None, features=None, *args, **kwargs):

            if features is not None:
                features['messages'] = [{
                    'role': Value(dtype='string', id=None),
                    'content': Value(dtype='string', id=None)
                }]
                features['images'] = [{'bytes': Value(dtype='binary', id=None), 'path': Value(dtype='string', id=None)}]
            ArrowWriter.__origin_init__(self, schema, features, *args, **kwargs)

        ArrowWriter.__origin_init__ = ArrowWriter.__init__
        ArrowWriter.__init__ = _new_init
        try:
            yield
        finally:
            ArrowWriter.__init__ = ArrowWriter.__origin_init__
            del ArrowWriter.__origin_init__

    def _cast_pil_image(self, dataset):
        features = dataset.features
        if 'images' in features and isinstance(features['images'], Image) and features['images'].decode:
            dataset = dataset.cast_column('images', Image(decode=False))
        return dataset

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        strict: bool = False,
        load_from_cache_file: bool = False,
        batch_size: int = 1000,
    ) -> DATASET_TYPE:
        from ..utils import sample_dataset
        if self.dataset_sample is not None:
            dataset = sample_dataset(dataset, self.dataset_sample, self.random_state)

        dataset = self.safe_rename_columns(dataset, self.columns_mapping)
        dataset = self.prepare_dataset(dataset)
        dataset = self._cast_pil_image(dataset)
        map_kwargs = {}
        if isinstance(dataset, HfDataset):
            map_kwargs.update({'num_proc': num_proc, 'load_from_cache_file': load_from_cache_file})
        with self._patch_arrow_writer():
            try:
                dataset_mapped = dataset.map(
                    self.batched_preprocess,
                    batched=True,
                    batch_size=batch_size,
                    fn_kwargs={'strict': strict},
                    remove_columns=list(dataset.features.keys()),
                    **map_kwargs)
            except NotImplementedError:
                pass
        if isinstance(dataset_mapped, HfDataset) and len(dataset) != len(dataset_mapped):
            logger.info(
                f'Dataset filtered, origin length: {len(dataset)}, filtered dataset length: {len(dataset_mapped)}')

        return dataset_mapped


class ResponsePreprocessor(RowPreprocessor):
    """Dataset compatible with older versions of ms-swift"""

    def __init__(self, *, columns_mapping: Optional[Dict[str, str]] = None, **kwargs) -> None:
        super().__init__(columns_mapping=columns_mapping, **kwargs)
        system_keys = ['system', 'system_prompt']
        query_keys = ['query', 'prompt', 'input', 'instruction', 'question']
        response_keys = ['response', 'answer', 'output', 'targets', 'target', 'answer_key', 'solution', 'answers'
                         ] + ['text', 'completion', 'content']
        for key in system_keys:
            self.columns_mapping[key] = 'system'
        for key in query_keys:
            self.columns_mapping[key] = 'query'
        for key in response_keys:
            self.columns_mapping[key] = 'response'

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = row.pop('response', None)
        if response is not None:
            if isinstance(response, (list, tuple)):
                # sometimes response is a list, pick one randomly
                response = self.random_state.choice(response)
        history = row.pop('history', None) or []
        query = row.pop('query', None)
        system = row.pop('system', None)
        if isinstance(history, str):  # e.g. "[['query1', 'response1']]"
            history = ast.literal_eval(history)
        history.append([query, response])

        row.update({'messages': history_to_messages(history, system)})
        return row


class AlpacaPreprocessor(ResponsePreprocessor):

    def __init__(self,
                 *,
                 concat_inst_input: Union[Callable[[str, str], str]] = '\n',
                 columns_mapping: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        """Alpaca format preprocessor

        Args:
            concat_inst_input: The concat sep between instruction and input
        """
        super().__init__(columns_mapping=columns_mapping, **kwargs)
        self.concat_inst_input = concat_inst_input

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        instruction = row.pop('instruction', None)
        input_ = row.pop('input', None)
        output = row.pop('output', None)
        if output is not None:
            row['response'] = output

        if instruction is not None or input_ is not None:
            instruction = instruction or ''
            input_ = input_ or ''
            if isinstance(self.concat_inst_input, str):
                query = instruction + self.concat_inst_input + input_
            else:
                query = self.concat_inst_input(instruction, input_)
            row['query'] = query
        return super().preprocess(row)


def default_repair_messages(s: Union[str, Any]) -> Any:
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s


class MessagesPreprocessor(RowPreprocessor):

    def __init__(
            self,
            *,
            # If set to None, automatic matching will be performed.
            role_key: Optional[str] = None,  # 'role', 'from'
            content_key: Optional[str] = None,  # 'content', 'value'
            user_role: Optional[str] = None,  # 'user', 'human'
            assistant_role: Optional[str] = None,  # 'assistant', 'gpt', 'bot'
            system_role: str = 'system',
            tool_role: str = 'tool',
            # 'conversation', 'conversations' -> 'messages'
            columns_mapping: Optional[Dict[str, str]] = None,
            repair_messages: Callable[[Union[str, List[Dict[str, str]]]],
                                      Optional[List[Dict[str, str]]]] = default_repair_messages,
            inner_key: Optional[str] = None,
            **kwargs):
        super().__init__(columns_mapping=columns_mapping, **kwargs)
        self.role_keys = ['role', 'from'] if role_key is None else [role_key]
        self.content_keys = ['content', 'value'] if content_key is None else [content_key]
        self.user_roles = ['user', 'human'] if user_role is None else [user_role]
        self.assistant_roles = ['assistant', 'gpt', 'bot'] if assistant_role is None else [assistant_role]

        self.system_role = system_role
        self.tool_role = tool_role
        self.repair_messages = repair_messages
        self.inner_key = inner_key

        message_keys = ['messages', 'conversation', 'conversations']
        for key in message_keys:
            self.columns_mapping[key] = 'messages'
        # sharegptq
        system_keys = ['system', 'system_prompt']
        if system_role not in system_keys:
            system_keys.append(system_role)
        for key in system_keys:
            self.columns_mapping[key] = 'system'

    @staticmethod
    def _is_sharegpt_format(message: Dict[str, str]) -> bool:
        if 'role' in message or 'content' in message:
            return False
        return True

    def sharegpt_to_messages(self, messages: List[Dict[str, str]], system: Optional[str]) -> List[Dict[str, str]]:
        self._to_std_key(messages, 'user', self.user_roles)
        self._to_std_key(messages, 'assistant', self.assistant_roles)
        new_messages = []
        if system is not None:
            new_messages.append({'role': 'system', 'content': system})
        for message in messages:
            if self.tool_role in message:
                user_message = {'role': 'tool', 'content': message[self.tool_role]}
            else:
                user_message = {'role': 'user', 'content': message['user']}
            assistant_message = {'role': 'assistant', 'content': message['assistant']}
            new_messages.append(user_message)
            new_messages.append(assistant_message)
        return new_messages

    def to_std_messages(self, messages: List[Dict[str, str]]) -> None:
        start_idx = 0
        if messages[0]['role'] == self.system_role:
            messages[0]['role'] = 'system'
            start_idx = 1
        for user_message, assistant_message in zip(messages[start_idx::2], messages[start_idx + 1::2]):
            user_role = user_message['role']
            assistant_role = assistant_message['role']
            if user_role in self.user_roles:
                user_message['role'] = 'user'
            elif user_role == self.tool_role:
                user_message['role'] = 'tool'
            if assistant_role in self.assistant_roles:
                assistant_message['role'] = 'assistant'

    @staticmethod
    def _to_std_key(messages: List[Dict[str, str]], std_key: str, optional_keys: List[str]) -> None:
        for message in messages:
            for key in optional_keys:
                if key in message:
                    message[std_key] = message.pop(key)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if 'rejected_messages' in row:
            row['rejected_messages'] = MessagesPreprocessor.preprocess(
                self, {'messages': row['rejected_messages']})['messages']
        messages = row['messages']
        if self.inner_key is not None:
            messages = messages[self.inner_key]
        messages: Optional[List[Dict[str, str]]] = self.repair_messages(messages)
        if not messages or isinstance(messages, str):
            return
        self._to_std_key(messages, 'role', self.role_keys)
        self._to_std_key(messages, 'content', self.content_keys)
        if self._is_sharegpt_format(messages[0]):
            system = row.pop('system', None)
            messages = self.sharegpt_to_messages(messages, system)
        else:
            self.to_std_messages(messages)  # inplace
        row['messages'] = messages
        return row


class YoloPreprocessor(RowPreprocessor):
    def __init__(self,
                 *,
                 columns_mapping: Optional[Dict[str, str]] = None,
                 dataset_sample: Optional[int] = None,
                 random_state: Optional[np.random.RandomState] = None,
                 traceback_limit: int = 10,
                 batch_size: int = 100,
                 num_proc: int = 1) -> None:
        # 调用父类的初始化方法
        super().__init__(columns_mapping=columns_mapping,
                        dataset_sample=dataset_sample,
                        random_state=random_state,
                        traceback_limit=traceback_limit)
        self.batch_size = batch_size
        self.num_proc = num_proc

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 复制输入的行数据，避免修改原始数据
        processed_row = dict(row)
        # 处理 image 字段
        self._cast_images(processed_row)
        image_path = processed_row.get('image')
        if image_path:
            if not os.path.exists(image_path):
                # 如果图像文件路径不存在，返回 None，该数据将被过滤
                return None
        # 处理 label 字段
        label_path = processed_row.get('label_path')
        if label_path:
            if os.path.exists(label_path):
                # 这里可以添加对标签文件的处理逻辑，例如读取标签文件内容
                # 假设标签文件是一个文本文件，每行包含一个标签
                # 每个标签格式为: 'label_id center_x center_y width height'
                with open(label_path, 'r') as label_file:
                    labels = label_file.readlines()
                bboxes = []
                for label in labels:
                    parts = label.strip().split()
                    label_id = int(parts[0])
                    classname = processed_row['class_names'][label_id]
                    yolo_bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                    bboxes.append([classname, yolo_bbox])
                # 将 bbox 转换为字符串以便兼容 pyarrow
                processed_row['bbox'] = [f"{bbox[0]}:{','.join(map(str, bbox[1]))}" for bbox in bboxes]
            else:
                # 如果标签文件路径不存在，返回 None，该数据将被过滤
                return None
        
        # 添加一个标记，用于后期处理
        processed_row['yolo'] = 1 
        return processed_row

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        # 调用父类的 prepare_dataset 方法
        dataset = super().prepare_dataset(dataset)
        # 可以添加其他对数据集的准备工作，例如过滤无效数据
        return dataset

    def batched_preprocess(self, batched_row: Dict[str, List[Any]], *, strict: bool) -> Dict[str, List[Any]]:
        processed_batched_row = {}
        for key in batched_row:
            processed_batched_row[key] = []
        for i in range(len(batched_row[list(batched_row.keys())[0]])):
            row = {k: v[i] for k, v in batched_row.items()}
            processed_row = self.preprocess(row)
            if processed_row is not None:
                for key in processed_row:
                    if key not in processed_batched_row:
                        processed_batched_row[key] = []
                    processed_batched_row[key].append(processed_row[key])
        return processed_batched_row

    def __call__(self, dataset: HfDataset, **kwargs) -> HfDataset:
        # 重写 __call__ 方法，绕过父类的 __call__ 方法
        # 首先对数据集进行安全的列重命名
        dataset = self.safe_rename_columns(dataset, self.columns_mapping)
        # 调用 prepare_dataset 进行数据集准备
        dataset = self.prepare_dataset(dataset)
        dataset = self._cast_pil_image(dataset)
        map_kwargs = {}
        if self.num_proc > 1:
            map_kwargs['num_proc'] = self.num_proc
        # 对数据集的每一批次调用 batched_preprocess 方法进行处理
        dataset_mapped = dataset.map(
            self.batched_preprocess,
            batched=True,
            batch_size=self.batch_size,
            fn_kwargs={'strict': kwargs.get('strict', False)},
            remove_columns=list(dataset.features.keys()),
            **map_kwargs
        )
        return dataset_mapped

class AutoPreprocessor:

    def __init__(self, *, columns_mapping: Optional[Dict[str, str]] = None, **kwargs) -> None:
        self.columns_mapping = columns_mapping or {}
        self.kwargs = kwargs

    def _get_preprocessor(self, dataset: DATASET_TYPE) -> RowPreprocessor:
        features = dataset.features
        for key in ['conversation', 'conversations', 'messages']:
            if key in features:
                return MessagesPreprocessor(**self.kwargs)
        if 'instruction' in features and 'input' in features:
            return AlpacaPreprocessor(**self.kwargs)
        if 'yolo' in features:
            return YoloPreprocessor(**self.kwargs)
        return ResponsePreprocessor(**self.kwargs)

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        strict: bool = False,
        load_from_cache_file: bool = False,
    ) -> DATASET_TYPE:
        dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns_mapping)
        preprocessor = self._get_preprocessor(dataset)
        return preprocessor(dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)
