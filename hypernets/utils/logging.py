"""Logging utilities."""

# pylint: disable=unused-import
# pylint: disable=g-bad-import-order
# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging as _logging
import os as _os
import sys as _sys
import traceback as _traceback
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN

# settings
_LOG_NAME = _os.environ.get('HYN_LOG_NAME', 'hypernets')
_LOG_LEVEL = _os.environ.get('HYN_LOG_LEVEL', 'INFO')
_LOG_FORMAT = _os.environ.get(
    'HYN_LOG_FORMAT',
    # '%(name)s %(levelname).1s%(asctime)s.%(msecs)d %(filename)s %(lineno)d - %(message)s'
    '%(levelname).1s %(sname)s.%(filename)s %(lineno)d - %(message)s'
    # '%(module)s %(levelname).1s %(filename)s %(lineno)d - %(message)s'
)
# _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DATE_FORMAT = _os.environ.get('HYN_LOG_DATE_FORMAT', '%m%d %H:%M:%S')


class MyLogFormatter(_logging.Formatter):

    def __init__(self, fmt=None, datefmt=None, style='%'):
        super(MyLogFormatter, self).__init__(fmt, datefmt, style)

        self.with_simple_name = fmt.find(style + '(sname)') >= 0

    def formatMessage(self, record):
        if self.with_simple_name:
            record.sname = self.get_simple_name(record.name)
        return super(MyLogFormatter, self).formatMessage(record)

    @staticmethod
    def get_simple_name(name):
        sa = name.split('.')
        if len(sa) <= 1:
            return name

        names = [sa[0]] + \
                [sa[i][0] for i in range(1, len(sa) - 1)]
        return '.'.join(names)


class MyLogger(_logging.getLoggerClass()):
    FATAL = FATAL
    ERROR = ERROR
    INFO = INFO
    DEBUG = DEBUG
    WARN = WARN

    def __init__(self, name, level=_LOG_LEVEL) -> None:
        super(MyLogger, self).__init__(name, level)

        self.findCaller = _logger_find_caller
        self.setLevel(_LOG_LEVEL)

        # Don't further configure the TensorFlow logger if the root logger is
        # already configured. This prevents double logging in those cases.
        if not _logging.getLogger().handlers:
            # Determine whether we are in an interactive environment
            _interactive = False
            try:
                # This is only defined in interactive shells.
                if _sys.ps1: _interactive = True
            except AttributeError:
                # Even now, we may be in an interactive shell with `python -i`.
                _interactive = _sys.flags.interactive

            # If we are in an interactive environment (like Jupyter), set loglevel
            # to INFO and pipe the output to stdout.
            if _interactive:
                _logging_target = _sys.stdout
            else:
                _logging_target = _sys.stderr

            # Add the output handler.
            _handler = _logging.StreamHandler(_logging_target)
            # _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
            _handler.setFormatter(MyLogFormatter(_LOG_FORMAT, _DATE_FORMAT))
            self.addHandler(_handler)

    def log(self, level, msg, *args, **kwargs):
        super(MyLogger, self).log(level, msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        self.log(FATAL, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.log(ERROR, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log(WARN, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(INFO, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.log(DEBUG, msg, *args, **kwargs)

    def log_if(self, level, msg, condition, *args):
        """Log 'msg % args' at level 'level' only if condition is fulfilled."""
        if condition:
            self.log(level, msg, *args)

    def log_every_n(self, level, msg, n, *args):
        """Log 'msg % args' at level 'level' once per 'n' times.

        Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.
        Not threadsafe.

        Args:
          level: The level at which to log.
          msg: The message to be logged.
          n: The number of times this should be called before it is logged.
          *args: The args to be substituted into the msg.
        """
        count = _get_next_log_count_per_token(_get_file_and_line())
        self.log_if(level, msg, not (count % n), *args)

    def log_first_n(self, level, msg, n, *args):  # pylint: disable=g-bad-name
        """Log 'msg % args' at level 'level' only first 'n' times.

        Not threadsafe.

        Args:
          level: The level at which to log.
          msg: The message to be logged.
          n: The number of times this should be called before it is logged.
          *args: The args to be substituted into the msg.
        """
        count = _get_next_log_count_per_token(_get_file_and_line())
        self.log_if(level, msg, count < n, *args)

    def is_debug_enabled(self):
        return self.isEnabledFor(DEBUG)

    def is_info_enabled(self):
        return self.isEnabledFor(INFO)

    def is_warning_enabled(self):
        return self.isEnabledFor(WARN)


_logging.setLoggerClass(MyLogger)


def get_logger(name=_LOG_NAME):
    return _logging.getLogger(name)


# compatible with pylog
def getLogger(name=_LOG_NAME):
    return _logging.getLogger(name)


def get_level():
    """Return how much logging output will be produced for newer logger."""
    return _LOG_LEVEL


def set_level(v):
    """Sets newer logger threshold for what messages will be logged."""
    global _LOG_LEVEL
    _LOG_LEVEL = v


def _get_caller(offset=3):
    """Returns a code and frame object for the lowest non-logging stack frame."""
    # Use sys._getframe().  This avoids creating a traceback object.
    # pylint: disable=protected-access
    f = _sys._getframe(offset)
    # pylint: enable=protected-access
    our_file = f.f_code.co_filename
    f = f.f_back
    while f:
        code = f.f_code
        if code.co_filename != our_file:
            return code, f
        f = f.f_back
    return None, None


# The definition of `findCaller` changed in Python 3.2,
# and further changed in Python 3.8
if _sys.version_info.major >= 3 and _sys.version_info.minor >= 8:

    def _logger_find_caller(stack_info=False, stacklevel=1):  # pylint: disable=g-wrong-blank-lines
        # code, frame = _get_caller(4)
        code, frame = _get_caller(4)
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return '(unknown file)', 0, '(unknown function)', sinfo
elif _sys.version_info.major >= 3 and _sys.version_info.minor >= 2:

    def _logger_find_caller(stack_info=False):  # pylint: disable=g-wrong-blank-lines
        code, frame = _get_caller(4)
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return '(unknown file)', 0, '(unknown function)', sinfo
else:
    def _logger_find_caller():  # pylint: disable=g-wrong-blank-lines
        code, frame = _get_caller(4)
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name)
        else:
            return '(unknown file)', 0, '(unknown function)'

# Counter to keep track of number of log entries per token.
_log_counter_per_token = {}


def _get_next_log_count_per_token(token):
    """Wrapper for _log_counter_per_token.

    Args:
      token: The token for which to look up the count.

    Returns:
      The number of times this function has been called with
      *token* as an argument (starting at 0)
    """
    global _log_counter_per_token  # pylint: disable=global-variable-not-assigned
    _log_counter_per_token[token] = 1 + _log_counter_per_token.get(token, -1)
    return _log_counter_per_token[token]


def _get_file_and_line():
    """Returns (filename, linenumber) for the stack frame."""
    code, f = _get_caller()
    if not code:
        return '<unknown>', 0
    return code.co_filename, f.f_lineno
