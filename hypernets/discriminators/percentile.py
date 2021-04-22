# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.utils.logging import get_logger

from ._base import BaseDiscriminator, get_percentile_score

logger = get_logger(__name__)


class PercentileDiscriminator(BaseDiscriminator):
    def __init__(self, percentile, min_trials=5, min_steps=5, stride=1, history=None, optimize_direction='min'):
        BaseDiscriminator.__init__(self, min_trials, min_steps, stride, history, optimize_direction)
        assert percentile >= 0.0 and percentile <= 100.0, f'Percentile which must be between 0 and 100 inclusive. got {percentile}'
        self.percentile = percentile

    def _is_promising(self, iteration_trajectory, group_id):
        n_step = len(iteration_trajectory) - 1
        percentile_score = get_percentile_score(self.history, n_step, group_id, self.percentile, self._sign)
        current_trial_score = iteration_trajectory[-1]
        result = current_trial_score * self._sign > percentile_score * self._sign
        logger.info(
            f'is promising:{result}, percentile_score:{percentile_score}, current_trial_score:{current_trial_score}')
        return result


class ProgressivePercentileDiscriminator(BaseDiscriminator):
    def __init__(self, percentile_list, min_trials=5, min_steps=5, stride=1, history=None, optimize_direction='min'):
        BaseDiscriminator.__init__(self, min_trials, min_steps, stride, history, optimize_direction)
        assert len(percentile_list) > 0, 'percentile list has at least one element'
        assert all([percentile >= 0.0 and percentile <= 100.0 for percentile in
                    percentile_list]), f'Percentile which must be between 0 and 100 inclusive. got {percentile_list}'
        self.percentile_list = percentile_list

    def _is_promising(self, iteration_trajectory, group_id):
        n_step = len(iteration_trajectory) - 1
        progress = (n_step - self.min_steps) // self.stride

        if progress < len(self.percentile_list):
            percentile = self.percentile_list[progress]
        else:
            percentile = self.percentile_list[-1]

        percentile_score = get_percentile_score(self.history, n_step, group_id, percentile, self._sign)
        current_trial_score = iteration_trajectory[-1]
        result = current_trial_score * self._sign > percentile_score * self._sign
        logger.info(
            f'progress:{progress}, percentile:{percentile}, is promising:{result}, percentile_score:{percentile_score}, current_trial_score:{current_trial_score}')
        return result
