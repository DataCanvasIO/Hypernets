# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import numpy as np

_hypernets_random_state = None


def set_random_state(seed):
    global _hypernets_random_state
    if seed is None:
        _hypernets_random_state = None
    else:
        _hypernets_random_state = np.random.RandomState(seed=seed)


def get_random_state():
    global _hypernets_random_state
    if _hypernets_random_state is None:
        return np.random.RandomState()
    else:
        return _hypernets_random_state


def randint():
    return get_random_state().randint(0, 65535)
