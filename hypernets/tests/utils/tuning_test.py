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


def func_early_stopping(p1=Choice(['a', 'b'], random_state=np.random.RandomState(9527)),
                        p2=Int(1, 10, 2, random_state=np.random.RandomState(9527)),
                        p3=Real(1.0, 5.0, random_state=np.random.RandomState(9527)),
                        p4=9):
    print(f'p1:{p1},p2:{p2},p3{p3},p4:{p4}')
    return 0.6


class Test_ParamTuning():
    def test_search_params(self):
        print('start')
        history = search_params(func1, 'grid', max_trials=10, optimize_direction='max')
        best = history.get_best()
        assert best.reward[0] == 14.370000000000001
        assert best.trial_no == 10

    def test_trigger_by_trials(self):
        from hypernets.core import EarlyStoppingCallback
        es = EarlyStoppingCallback(3, 'max',
                                   time_limit=3600,
                                   expected_reward=1)

        history = search_params(func_early_stopping, 'grid', max_trials=10, optimize_direction='max', callbacks=[es])
        best = history.get_best()
        assert best.reward[0] == 0.6
        assert best.trial_no == 1
        assert len(history.trials) == 4
        assert es.triggered_reason == EarlyStoppingCallback.REASON_TRIAL_LIMIT

    def test_trigger_by_reward(self):
        from hypernets.core import EarlyStoppingCallback
        es = EarlyStoppingCallback(3, 'max',
                                   time_limit=3600,
                                   expected_reward=0.5)

        history = search_params(func_early_stopping, 'grid', max_trials=10, optimize_direction='max', callbacks=[es])
        best = history.get_best()
        assert best.reward[0] == 0.6
        assert best.trial_no == 1
        assert len(history.trials) == 1
        assert es.triggered_reason == EarlyStoppingCallback.REASON_EXPECTED_REWARD

