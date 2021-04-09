import inspect
import sys as _sys
import time
import traceback as _traceback
from collections import Iterable

from . import logging


def tic_toc(log_level='info', name=None, details=True):
    log_level = logging.to_level(log_level)

    def decorate(fn):
        assert callable(fn)

        if log_level < logging.get_level():
            return fn

        logger_name = name if name is not None else f'[tic-toc] {fn.__name__} @'
        logger = logging.get_logger(logger_name)
        logger.findCaller = _logger_find_caller

        fn_sig = inspect.signature(fn) if details else None

        def tic_toc_call(*args, **kwargs):
            tic = time.time()
            r = fn(*args, **kwargs)
            toc = time.time()

            msg = f'elapsed {toc - tic:.3f} seconds'
            if details and (len(args) > 0 or len(kwargs) > 0):
                ba = fn_sig.bind(*args, **kwargs)
                args = [f'{type(v).__name__}<{k}>' if k == 'self' else f'{k}={_format_value(v)}'
                        for k, v in ba.arguments.items()]
                msg += f', details:\t{", ".join(args)}'

            logger.log(log_level, msg)

            return r

        return tic_toc_call

    return decorate


_VALUE_LEN_LIMIT = 10


def _format_value(v):
    if v is None or isinstance(v, (int, float, bool, complex)):
        r = v
    elif isinstance(v, str):
        if len(v) > _VALUE_LEN_LIMIT:
            r = v[:_VALUE_LEN_LIMIT]
            r = f'{r}...[len={len(v)}]'
        else:
            r = v
    elif isinstance(v, (bytes, bytearray)):
        r = f'{type(v).__name__}[{len(v)}]'
    elif isinstance(v, type):
        r = v.__name__
    elif hasattr(v, 'shape'):
        r = f'{type(v).__name__}[shape={getattr(v, "shape")}]'
    elif hasattr(v, '__name__'):
        r = f'{type(v).__name__}[name={getattr(v, "__name__")}]'
    elif isinstance(v, dict):
        r = type(v)()
        for k, v in v.items():
            r[_format_value(k)] = _format_value(v)
            if len(r) >= _VALUE_LEN_LIMIT:
                r['...'] = '...'
                break
    elif isinstance(v, Iterable):
        r = []
        for e in v:
            r.append(_format_value(e))
            if len(r) >= _VALUE_LEN_LIMIT:
                r.append('...')
                break
        r = type(v).__name__, r
    else:
        r = f'{type(v).__name__}'

    return r


###############################################################
# adapted from logging


# The definition of `findCaller` changed in Python 3.2,
# and further changed in Python 3.8
if _sys.version_info.major >= 3 and _sys.version_info.minor >= 8:

    def _logger_find_caller(stack_info=False, stacklevel=1):  # pylint: disable=g-wrong-blank-lines
        code, frame = logging._get_caller(6)
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return '(unknown file)', 0, '(unknown function)', sinfo
elif _sys.version_info.major >= 3 and _sys.version_info.minor >= 2:

    def _logger_find_caller(stack_info=False):  # pylint: disable=g-wrong-blank-lines
        code, frame = logging._get_caller(5)
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return '(unknown file)', 0, '(unknown function)', sinfo
else:
    def _logger_find_caller():  # pylint: disable=g-wrong-blank-lines
        code, frame = logging._get_caller(6)
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name)
        else:
            return '(unknown file)', 0, '(unknown function)'
