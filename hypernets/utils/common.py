# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import cProfile
import contextlib
import inspect
import math
import os
import pstats
import re
import tempfile
import uuid
from collections import OrderedDict

from . import logging

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
                      if p.name != 'self']  # and p.kind != p.VAR_KEYWORD]
        return parameters

    fn_get_params = getattr(obj, 'get_params', None)
    if callable(fn_get_params):
        return fn_get_params()

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
        return f'{type(e).__name__}:{e}, at <to_repr>: {type(obj).__name__}'


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


@contextlib.contextmanager
def context(msg):
    # Stolen from https://stackoverflow.com/a/17677938/356729
    try:
        yield
    except Exception as ex:
        if ex.args:
            msg = u'{}: {}'.format(msg, ex.args[0])
        else:
            msg = str(msg)
        ex.args = (msg,) + ex.args[1:]
        raise


def profile(fn, sort_by='cumtime'):
    assert callable(fn)

    p = cProfile.Profile()
    p.enable()
    fn()
    p.disable()
    s = pstats.Stats(p).sort_stats(sort_by)
    return s


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


def get_temp_file_path(prefix=None, suffix=None):
    fd, file_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    os.remove(file_path)
    return file_path


def get_temp_dir_path(prefix=None, suffix=None, create=True):
    file_path = tempfile.mkdtemp(prefix=prefix, suffix=suffix)
    if not create:
        os.rmdir(file_path)  # empty dir
    return file_path


_SHORT_UUID_CHARS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g",
                     "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
                     "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
                     "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def generate_short_id():
    long_id = str(uuid.uuid4()).replace("-", '')
    buffer = []
    for i in range(0, 8):
        start = i * 4
        end = i * 4 + 4
        val = int(long_id[start:end], 16)
        buffer.append(_SHORT_UUID_CHARS[val % 62])
    return "".join(buffer)
