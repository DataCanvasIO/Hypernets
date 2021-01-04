# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import math
import os
import sys
import uuid


def generate_id():
    return str(uuid.uuid1())


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


def config(key, default=None):
    # parse config from command line
    argv = sys.argv
    key_alias = key.replace('_', '-')
    accept_items = {f'--{key}', f'-{key}', f'--{key_alias}', f'-{key_alias}'}
    for i in range(len(argv) - 1):
        if argv[i] in accept_items:
            return argv[i + 1]

    # parse config from environs
    return os.environ.get(f'HYN_{key}'.upper(), default)


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
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except:
        return False