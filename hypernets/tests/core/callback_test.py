# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.core.callbacks import EarlyStoppingCallback, EarlyStoppingError
import pytest


class Test_Callback:
    def test_early_stopping(self):
        es = EarlyStoppingCallback(3, 'min')
        es.on_trial_end(None, None, 1, [0.9], True, 0)
        es.on_trial_end(None, None, 2, [0.9], True, 0)
        es.on_trial_end(None, None, 3, [0.9], True, 0)

        with pytest.raises(EarlyStoppingError) as ese:
            es.on_trial_end(None, None, 4, [0.9], True, 0)
        assert ese.value.args[0].find('reason: max_no_improvement_trials') > -0

        es = EarlyStoppingCallback(3, 'min')
        es.on_trial_end(None, None, 1, [0.9], True, 0)
        es.on_trial_end(None, None, 2, [0.8], True, 0)
        es.on_trial_end(None, None, 3, [0.8], True, 0)
        es.on_trial_end(None, None, 4, [0.8], True, 0)

        with pytest.raises(EarlyStoppingError) as ese:
            es.on_trial_end(None, None, 5, [0.8], True, 0)
        assert ese.value.args[0].find('reason: max_no_improvement_trials') > -0

        es = EarlyStoppingCallback(3, 'max')
        es.on_trial_end(None, None, 1, [0.9], True, 0)
        es.on_trial_end(None, None, 2, [0.9], True, 0)
        es.on_trial_end(None, None, 3, [0.9], True, 0)

        with pytest.raises(EarlyStoppingError) as ese:
            es.on_trial_end(None, None, 4, [0.9], True, 0)
        assert ese.value.args[0].find('reason: max_no_improvement_trials') > -0

        es = EarlyStoppingCallback(3, 'max')
        es.on_trial_end(None, None, 1, [0.9], True, 0)
        es.on_trial_end(None, None, 2, [0.91], True, 0)
        es.on_trial_end(None, None, 3, [0.91], True, 0)
        es.on_trial_end(None, None, 4, [0.91], True, 0)

        with pytest.raises(EarlyStoppingError) as ese:
            es.on_trial_end(None, None, 5, [0.91], True, 0)
        assert ese.value.args[0].find('reason: max_no_improvement_trials') > -0

#    def test_early_stopping_by_pareto(self):
#
#        es = EarlyStoppingCallback(3, 'min')
#        es.on_trial_end(None, None, 1, [0.9, 0.9], True, 0)
#        es.on_trial_end(None, None, 2, [0.95, 0.8], True, 0)
#        es.on_trial_end(None, None, 3, [0.95, 0.7], True, 0)
#
#        with pytest.raises(EarlyStoppingError) as ese:
#            es.on_trial_end(None, None, 4, [0.9, 0.9], True, 0)
#        assert ese.value.args[0].find('reason: max_no_improvement_trials') > -0
#
#        es = EarlyStoppingCallback(3, 'max')
#        es.on_trial_end(None, None, 1, [0.9, 0.9], True, 0)
#        es.on_trial_end(None, None, 2, [0.8, 0.95], True, 0)
#        es.on_trial_end(None, None, 3, [0.9, 0.9], True, 0)
#
#        with pytest.raises(EarlyStoppingError) as ese:
#            es.on_trial_end(None, None, 4, [0.9, 0.9], True, 0)
#        assert ese.value.args[0].find('reason: max_no_improvement_trials') > -0
