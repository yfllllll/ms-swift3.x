# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
import random
import re
import subprocess
import sys
import time
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Type, TypeVar)

import numpy as np
import torch.distributed as dist
from transformers import HfArgumentParser, enable_full_determinism, set_seed

from .logger import get_logger
from .np_utils import stat_array
from .torch_utils import broadcast_string, is_dist

logger = get_logger()


def check_json_format(obj: Any) -> Any:
    if obj is None or isinstance(
            obj, (int, float, str, complex)):  # bool is a subclass of int
        return obj

    if isinstance(obj, Sequence):
        res = []
        for x in obj:
            res.append(check_json_format(x))
    elif isinstance(obj, Mapping):
        res = {}
        for k, v in obj.items():
            res[k] = check_json_format(v)
    else:
        res = repr(obj)  # e.g. function
    return res


def _get_version(work_dir: str) -> int:
    if os.path.isdir(work_dir):
        fnames = os.listdir(work_dir)
    else:
        fnames = []
    v_list = [-1]
    for fname in fnames:
        m = re.match(r'v(\d+)', fname)
        if m is None:
            continue
        v = m.group(1)
        v_list.append(int(v))
    return max(v_list) + 1


def seed_everything(seed: Optional[int] = None,
                    full_determinism: bool = False,
                    *,
                    verbose: bool = True) -> int:

    if seed is None:
        seed_max = np.iinfo(np.int32).max
        seed = random.randint(0, seed_max)

    if full_determinism:
        enable_full_determinism(seed)
    else:
        set_seed(seed)
    if verbose:
        logger.info(f'Global seed set to {seed}')
    return seed


def add_version_to_work_dir(work_dir: str) -> str:
    """add version"""
    version = _get_version(work_dir)
    time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    sub_folder = f'v{version}-{time}'
    if dist.is_initialized() and is_dist():
        sub_folder = broadcast_string(sub_folder)
    work_dir = os.path.join(work_dir, sub_folder)
    return work_dir


_T = TypeVar('_T')


def parse_args(class_type: Type[_T],
               argv: Optional[List[str]] = None) -> Tuple[_T, List[str]]:
    parser = HfArgumentParser([class_type])
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) > 0 and argv[0].endswith('.json'):
        json_path = os.path.abspath(os.path.expanduser(argv[0]))
        args, = parser.parse_json_file(json_path)
        remaining_args = argv[1:]
    else:
        args, remaining_args = parser.parse_args_into_dataclasses(
            argv, return_remaining_strings=True)
    return args, remaining_args


def lower_bound(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    # The lower bound satisfying the condition "cond".
    while lo < hi:
        mid = (lo + hi) >> 1
        if cond(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def upper_bound(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    # The upper bound satisfying the condition "cond".
    while lo < hi:
        mid = (lo + hi + 1) >> 1  # lo + (hi-lo+1)>>1
        if cond(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def test_time(func: Callable[[], _T],
              number: int = 1,
              warmup: int = 0,
              timer: Optional[Callable[[], float]] = None) -> _T:
    # timer: e.g. time_synchronize
    timer = timer if timer is not None else time.perf_counter

    ts = []
    res = None
    # warmup
    for _ in range(warmup):
        res = func()

    for _ in range(number):
        t1 = timer()
        res = func()
        t2 = timer()
        ts.append(t2 - t1)

    ts = np.array(ts)
    _, stat_str = stat_array(ts)
    # print
    logger.info(f'time[number={number}]: {stat_str}')
    return res


def read_multi_line(addi_prompt: str = '') -> str:
    res = []
    prompt = f'<<<{addi_prompt} '
    while True:
        text = input(prompt) + '\n'
        prompt = ''
        res.append(text)
        if text.endswith('#\n'):
            res[-1] = text[:-2]
            break
    return ''.join(res)


def is_pai_training_job() -> bool:
    return 'PAI_TRAINING_JOB_ID' in os.environ


def get_pai_tensorboard_dir() -> Optional[str]:
    return os.environ.get('PAI_OUTPUT_TENSORBOARD')


def subprocess_run(command: List[str],
                   env: Optional[Dict[str, str]] = None,
                   stdout=None,
                   stderr=None) -> None:
    # stdoutm stderr: e.g. subprocess.PIPE.
    resp = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    resp.check_returncode()
    return resp
