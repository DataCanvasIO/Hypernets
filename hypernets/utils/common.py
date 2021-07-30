# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import hashlib
import inspect
import math
import pickle
import uuid
from collections import OrderedDict
from functools import partial
from io import BytesIO

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


def infer_task_type(y, excludes=None):
    assert excludes is None or isinstance(excludes, (list, tuple, set))

    if len(y.shape) > 1 and y.shape[-1] > 1:
        labels = list(range(y.shape[-1]))
        task = TASK_MULTILABEL  # 'multilable'
        return task, labels

    if hasattr(y, 'unique'):
        uniques = set(y.unique())
    elif isinstance(y, da.Array):
        uniques = set(da.unique(y).compute())
    else:
        uniques = set(y)

    if uniques.__contains__(np.nan):
        uniques.remove(np.nan)
    if excludes is not None and len(excludes) > 0:
        uniques -= set(excludes)
    n_unique = len(uniques)
    labels = []

    if n_unique == 2:
        logger.info(f'2 class detected, {uniques}, so inferred as a [binary classification] task')
        task = TASK_BINARY  # TASK_BINARY
        labels = sorted(uniques)
    else:
        if str(y.dtype).find('float') >= 0:
            logger.info(f'Target column type is {y.dtype}, so inferred as a [regression] task.')
            task = TASK_REGRESSION
        else:
            if n_unique > 1000:
                if str(y.dtype).find('int') >= 0:
                    logger.info('The number of classes exceeds 1000 and column type is {y.dtype}, '
                                'so inferred as a [regression] task ')
                    task = TASK_REGRESSION
                else:
                    raise ValueError('The number of classes exceeds 1000, please confirm whether '
                                     'your predict target is correct ')
            else:
                logger.info(f'{n_unique} class detected, inferred as a [multiclass classification] task')
                task = TASK_MULTICLASS
                labels = sorted(uniques)
    return task, labels


def hash_dataframe(df, method='md5', index=False):
    assert isinstance(df, (pd.DataFrame, dd.DataFrame))

    m = getattr(hashlib, method)()

    for col in df.columns:
        m.update(str(col).encode())

    if isinstance(df, dd.DataFrame):
        x = df.map_partitions(lambda part: pd.util.hash_pandas_object(part, index=index),
                              meta=(None, 'u8')).compute()
    else:
        x = pd.util.hash_pandas_object(df, index=index)

    np.vectorize(m.update, otypes=[None], signature='()->()')(x.values)

    return m.hexdigest()


def _hash_array(arr):
    if arr.shape[0] == 0:
        v = np.array([], dtype='u8').reshape((-1, 1))
    else:
        v = pd.util.hash_pandas_object(pd.DataFrame(arr), index=False).values.reshape((-1, 1))

    return v


def hash_array(arr, method='md5'):
    m = getattr(hashlib, method)()

    if isinstance(arr, da.Array):
        if len(arr.shape) == 1:
            arr = arr.compute_chunk_sizes().reshape(-1, 1)
        x = arr.map_blocks(_hash_array, dtype='u8').compute()
    elif isinstance(arr, np.ndarray):
        x = _hash_array(arr)
    else:
        x = _hash_array(np.array(arr))

    np.vectorize(m.update, otypes=[None], signature='()->()')(x)

    return m.hexdigest()


def hash_data(data, method='md5'):
    if isinstance(data, (pd.DataFrame, dd.DataFrame)):
        return hash_dataframe(data, method=method)
    elif isinstance(data, (pd.Series, dd.Series)):
        return hash_dataframe(data.to_frame(), method=method)
    elif isinstance(data, (np.ndarray, da.Array)):
        return hash_array(data, method=method)

    if isinstance(data, (bytes, bytearray)):
        pass
    elif isinstance(data, str):
        data = data.encode('utf-8')
    else:
        if isinstance(data, (list, tuple)):
            data = [hash_data(x) if x is not None else x for x in data]
        elif isinstance(data, dict):
            data = {hash_data(k): hash_data(v) if v is not None else v for k, v in data.items()}
        buf = BytesIO()
        pickle.dump(data, buf)
        data = buf.getvalue()
        buf.close()

    m = getattr(hashlib, method)()
    m.update(data)
    return m.hexdigest()


def load_module(mod_name):
    assert isinstance(mod_name, str) and mod_name.find('.') > 0

    cbs = mod_name.split('.')
    pkg, mod = '.'.join(cbs[:-1]), cbs[-1]
    pkg = __import__(pkg, fromlist=[''])
    mod = getattr(pkg, mod)
    return mod


def load_data(data, **kwargs):
    assert isinstance(data, (str, pd.DataFrame, dd.DataFrame))

    if isinstance(data, (pd.DataFrame, dd.DataFrame)):
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
