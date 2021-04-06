# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.utils.param_tuning import search_params
from hypernets.core.search_space import Choice, Real, Int
import numpy as np


def func1(p1=Choice(['a', 'b'], random_state=np.random.RandomState(9527)),
          p2=Int(1, 10, 2, random_state=np.random.RandomState(9527)),
          p3=Real(1.0, 5.0, random_state=np.random.RandomState(9527)), p4=9):
    print(f'p1:{p1},p2:{p2},p3{p3},p4:{p4}')
    return p2 * p3


class Test_ParamTuning():
    def test_search_params(self):
        print('start')
        history = search_params(func1, 'grid', max_trials=10, optimize_direction='max')
        best = history.get_best()
        assert best.reward == 14.370000000000001
        assert best.trial_no == 10
