# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.utils.logging import get_logger

from ._base import BaseDiscriminator, get_percentile_score
import numpy as np
logger = get_logger(__name__)


class PercentileDiscriminator(BaseDiscriminator):
    def __init__(self, percentile, min_trials=5, min_steps=5, stride=1, history=None, optimize_direction='min'):
        assert 0.0 <= percentile <= 100.0, f'Percentile which must be between 0 and 100 inclusive. got {percentile}'

        BaseDiscriminator.__init__(self, min_trials, min_steps, stride, history, optimize_direction)
        self.percentile = percentile

    def _is_promising(self, iteration_trajectory, group_id, end_iteration=None):
        n_step = len(iteration_trajectory) - 1
        percentile_score = get_percentile_score(self.history, n_step, group_id, self.percentile, self._sign)
        current_trial_score = iteration_trajectory[-1]
        result = current_trial_score * self._sign > percentile_score * self._sign
        if not result and logger.is_info_enabled():
            logger.info(f'direction:{self.optimize_direction}, promising:{result}, '
                        f'percentile_score:{percentile_score}, current_trial_score:{current_trial_score}, '
                        f'trajectory size:{n_step + 1}')
        return result

class OncePercentileDiscriminator(BaseDiscriminator):
    ## called when current iter is half of total iters
    def __init__(self, percentile, min_trials=5, min_steps=5, stride=1, history=None, optimize_direction='min'):
        assert 0.0 <= percentile <= 100.0, f'Percentile which must be between 0 and 100 inclusive. got {percentile}'

        BaseDiscriminator.__init__(self, min_trials, min_steps, stride, history, optimize_direction)
        self.percentile = percentile

    def get_previous_trials_scores(history, from_step, to_step, group_id):
        trial_scores = []
        for trial in history.trials:
            if not trial.succeeded:
                continue
            scores = trial.iteration_scores.get(group_id)
            if scores:
                trial_scores.append(scores[int(len(scores) / 2)])
        return np.array(trial_scores)

    def _is_promising(self, iteration_trajectory, group_id, end_iteration):
        result = True
        if len(iteration_trajectory) == int(end_iteration/2):
            n_step = int(end_iteration/2) - 1
            current_trial_score = iteration_trajectory[-1]
            self._sign = 1 if np.mean(iteration_trajectory[-5:]) > np.mean(iteration_trajectory[:5]) else -1
            self.optimize_direction = 'max' if self._sign > 0 else 'min'
            percentile_score = get_percentile_score(self.history, n_step, group_id, self.percentile, self._sign)
            result = current_trial_score * self._sign > percentile_score * self._sign
            if not result and logger.is_info_enabled():
                logger.info(f'direction:{self.optimize_direction}, promising:{result}, '
                            f'percentile_score:{percentile_score}, current_trial_score:{current_trial_score}, '
                            f'trajectory size:{n_step + 1}')
        return result
    
class ProgressivePercentileDiscriminator(BaseDiscriminator):
    def __init__(self, percentile_list, min_trials=5, min_steps=5, stride=1, history=None, optimize_direction='min'):
        assert len(percentile_list) > 0, 'percentile list has at least one element'
        assert all([0.0 <= percentile <= 100.0 for percentile in percentile_list]), \
            f'Percentile which must be between 0 and 100 inclusive. got {percentile_list}'

        BaseDiscriminator.__init__(self, min_trials, min_steps, stride, history, optimize_direction)
        self.percentile_list = percentile_list

    def _is_promising(self, iteration_trajectory, group_id, end_iteration=None):
        n_step = len(iteration_trajectory) - 1
        progress = (n_step - self.min_steps) // self.stride

        if progress < len(self.percentile_list):
            percentile = self.percentile_list[progress]
        else:
            percentile = self.percentile_list[-1]

        percentile_score = get_percentile_score(self.history, n_step, group_id, percentile, self._sign)
        current_trial_score = iteration_trajectory[-1]
        result = current_trial_score * self._sign > percentile_score * self._sign
        if not result and logger.is_info_enabled():
            logger.info(f'progress:{progress}, percentile:{percentile}, is promising:{result}, '
                        f'percentile_score:{percentile_score}, current_trial_score:{current_trial_score}, '
                        f'trajectory size:{n_step + 1}')
        return result
