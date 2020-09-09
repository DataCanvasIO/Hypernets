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
    accept_items = {f'--{key}', f'-{key}'}
    for i in range(len(argv) - 1):
        if argv[i] in accept_items:
            return argv[i + 1]

    # parse config from environs
    return os.environ.get(f'HYN_{key}'.upper(), default)
