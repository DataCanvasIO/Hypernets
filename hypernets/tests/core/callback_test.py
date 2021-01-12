# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.core.callbacks import EarlyStoppingCallback,EarlyStoppingError
import pytest

class Test_Callback:
    def test_early_stopping(self):
        es = EarlyStoppingCallback(3, 'min')
        es.on_trial_end(None, None, 1, 0.9, True, 0)
        es.on_trial_end(None, None, 2, 0.9, True, 0)
        es.on_trial_end(None, None, 3, 0.9, True, 0)

        with pytest.raises(EarlyStoppingError) as ese:
            es.on_trial_end(None, None, 4, 0.9, True, 0)
        assert ese.value.args[0] == 'Early stopping on trial : 4, best reward: 0.9, best_trial: 1'

        es = EarlyStoppingCallback(3, 'min')
        es.on_trial_end(None, None, 1, 0.9, True, 0)
        es.on_trial_end(None, None, 2, 0.8, True, 0)
        es.on_trial_end(None, None, 3, 0.8, True, 0)
        es.on_trial_end(None, None, 4, 0.8, True, 0)

        with pytest.raises(EarlyStoppingError) as ese:
            es.on_trial_end(None, None, 5, 0.8, True, 0)
        assert ese.value.args[0] == 'Early stopping on trial : 5, best reward: 0.8, best_trial: 2'

        es = EarlyStoppingCallback(3, 'max')
        es.on_trial_end(None, None, 1, 0.9, True, 0)
        es.on_trial_end(None, None, 2, 0.9, True, 0)
        es.on_trial_end(None, None, 3, 0.9, True, 0)

        with pytest.raises(EarlyStoppingError) as ese:
            es.on_trial_end(None, None, 4, 0.9, True, 0)
        assert ese.value.args[0] == 'Early stopping on trial : 4, best reward: 0.9, best_trial: 1'

        es = EarlyStoppingCallback(3, 'max')
        es.on_trial_end(None, None, 1, 0.9, True, 0)
        es.on_trial_end(None, None, 2, 0.91, True, 0)
        es.on_trial_end(None, None, 3, 0.91, True, 0)
        es.on_trial_end(None, None, 4, 0.91, True, 0)

        with pytest.raises(EarlyStoppingError) as ese:
            es.on_trial_end(None, None, 5, 0.91, True, 0)
        assert ese.value.args[0] == 'Early stopping on trial : 5, best reward: 0.91, best_trial: 2'
