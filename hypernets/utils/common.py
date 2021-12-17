# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import hashlib
import inspect
import math
import pickle
import uuid
import re
from collections import OrderedDict
from functools import partial
from io import BytesIO
import copy

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from . import logging
from .const import TASK_BINARY, TASK_REGRESSION, TASK_MULTICLASS, TASK_MULTILABEL

logger = logging.get_logger(__name__)


def generate_id():
    return str(uuid.uuid1())


def get_params(obj, include_default=False):
    def _get_init_params(cls):
        init = cls.__init__
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        return parameters

    out = OrderedDict()
    for p in _get_init_params(type(obj)):
        name = p.name
        value = getattr(obj, name, None)
        if include_default or value is not p.default:
            out[name] = value

    return out


def to_repr(obj, excludes=None):
    try:
        if excludes is None:
            excludes = []
        out = ['%s=%r' % (k, v) for k, v in get_params(obj).items() if k not in excludes]
        repr_ = ', '.join(out)
        return f'{type(obj).__name__}({repr_})'
    except Exception as e:
        if logger.is_info_enabled():
            logger.info(e)
        return f'{e} <to_repr>: {obj}'


def combinations(n, m_max, m_min=1):
    if m_max > n or m_max <= 0:
        m_max = n
    if m_min < 1:
        m_min = 1
    if m_min == 1 and m_max == n:
        return 2 ** n - 1
    else:
        sum = 0
        for i in range(m_min, m_max + 1):
            c = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
            sum += c
        return sum


class Counter(object):
    def __init__(self):
        from threading import Lock

        super(Counter, self).__init__()
        self._value = 0
        self._lock = Lock()

    @property
    def value(self):
        return self._value

    def __call__(self, *args, **kwargs):
        with self._lock:
            self._value += 1
            return self._value

    def inc(self, step=1):
        with self._lock:
            self._value += step
            return self._value

    def reset(self):
        with self._lock:
            self._value = 0
            return self._value


def isnotebook():
    '''code from https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    :return:
    '''
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except:
        return False


def load_module(mod_name):
    assert isinstance(mod_name, str) and mod_name.find('.') > 0

    cbs = mod_name.split('.')
    pkg, mod = '.'.join(cbs[:-1]), cbs[-1]
    pkg = __import__(pkg, fromlist=[''])
    mod = getattr(pkg, mod)
    return mod


def load_data(data, **kwargs):
    if not isinstance(data, str):
        if type(data).__name__.find('DataFrame') < 0:
            logger.warning(f'You data type {type(data).__name__} is not DataFrame.')
        return data

    import os.path as path
    import glob

    try:
        from dask.distributed import default_client as dask_default_client
        client = dask_default_client()
        dask_enabled, worker_count = True, len(client.ncores())
    except ValueError:
        dask_enabled, worker_count = False, 1

    fmt_mapping = {
        'csv': (partial(pd.read_csv, low_memory=False), dd.read_csv),
        'txt': (pd.read_csv, dd.read_csv),
        'parquet': (pd.read_parquet, dd.read_parquet),
        'par': (pd.read_parquet, dd.read_parquet),
        'json': (pd.read_json, dd.read_json),
        'pkl': (pd.read_pickle, None),
        'pickle': (pd.read_pickle, None),
    }

    def get_file_format(file_path):
        return path.splitext(file_path)[-1].lstrip('.')

    def get_file_format_by_glob(data_pattern):
        for f in glob.glob(data_pattern, recursive=True):
            fmt_ = get_file_format(f)
            if fmt_ in fmt_mapping.keys():
                return fmt_
        return None

    if glob.has_magic(data):
        fmt = get_file_format_by_glob(data)
    elif not path.exists(data):
        raise ValueError(f'Not found path {data}')
    elif path.isdir(data):
        path_pattern = f'{data}*' if data.endswith(path.sep) else f'{data}{path.sep}*'
        fmt = get_file_format_by_glob(path_pattern)
    else:
        fmt = path.splitext(data)[-1].lstrip('.')

    if fmt not in fmt_mapping.keys():
        # fmt = fmt_mapping.keys()[0]
        raise ValueError(f'Not supported data format{fmt}')
    fn = fmt_mapping[fmt][int(dask_enabled)]
    if fn is None:
        raise ValueError(f'Not supported data format{fmt}')

    if dask_enabled and path.isdir(data) and not glob.has_magic(data):
        data = f'{data}*' if data.endswith(path.sep) else f'{data}{path.sep}*'
    df = fn(data, **kwargs)

    if dask_enabled and worker_count > 1 and df.npartitions < worker_count:
        df = df.repartition(npartitions=worker_count)

    return df


def human_data_size(value):
    def r(v, unit):
        return "%s%s" % (round(v, 2), unit)

    if value < 1024 * 1024:
        return r(value / 1024, "KB")
    elif 1024 * 1024 < value <= 1024 * 1024 * 1024:
        return r(value / 1024 / 1024, "MB")
    else:
        return r(value / 1024 / 1024 / 1024, "GB")


def camel_to_snake(camel_str):
    """
        example:
            Convert 'camelToSnake' to 'camel_to_snake'
    """
    p = re.compile(r'([a-z]|\d)([A-Z])')
    sub = re.sub(p, r'\1_\2', camel_str).lower()
    return sub


def _recursion_replace(container):
    """Replace camel-case keys in *container* into snake-case
    Parameters
    ----------
    container: list, dict, required
    Returns
    -------
    """

    if isinstance(container, list):
        new_container = []
        for ele in container:
            if isinstance(ele, (list, dict)):
                new_ele = _recursion_replace(ele)
                new_container.append(new_ele)
            else:
                new_container.append(ele)
    elif isinstance(container, dict):
        new_container = {}
        for k, v in container.items():
            if isinstance(v, (dict, list)):
                snake_key_dict = _recursion_replace(v)
            else:
                snake_key_dict = v
            new_container[camel_to_snake(k)] = snake_key_dict  # attach to parent
    else:
        raise ValueError(f"Input is not a `dict` or `list`: {container}")

    return new_container


def camel_keys_to_snake(d: dict):
    """
    example:
        Convert dict:
            {
                'datasetConf': {
                    'trainData': ['./train.csv']
                }
            }
        to:
            {
                'dataset_conf': {
                    'train_data': ['./train.csv']
                }
            }
    """
    ret_dict = _recursion_replace(d)
    return ret_dict
