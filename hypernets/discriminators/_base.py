# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import numpy as np

from ..core import TrialHistory
from ..utils import to_repr


class UnPromisingTrial(Exception):
    pass


class BaseDiscriminator(object):
    """
    Discriminator is used to determine whether to continue training
    """

    def __init__(self, min_trials=5, min_steps=5, stride=1, history: TrialHistory = None, optimize_direction='min'):
        super(BaseDiscriminator, self).__init__()

        self.history = history
        self.min_trials = min_trials
        self.min_steps = min_steps
        self.stride = stride
        self.optimize_direction = optimize_direction.lower()
        if self.optimize_direction == 'min':
            self._sign = -1
        else:
            self._sign = 1

    def bind_history(self, history):
        self.history = history

    def is_promising(self, iteration_trajectory, group_id, end_iteration):
        if self.history is None:
            raise ValueError('`history` is not bound')
        n_step = len(iteration_trajectory)
        if n_step < self.min_steps:
            return True

        trial_scores = get_previous_trials_scores(self.history, n_step - 1, n_step - 1, group_id)
        if len(trial_scores) < self.min_trials:
            return True
        if self.stride > 1:
            if ((n_step - self.min_steps) % self.stride) > 0:
                return True

        return self._is_promising(iteration_trajectory, group_id, end_iteration)

    def _is_promising(self, iteration_trajectory, group_id, end_iteration):
        """
        discriminate whether continuing training a trial is promising

        Args:
            iteration_trajectory:
                list, The scores of each step in the iteration process are arranged from front to back
            group_id:
                Str, It is used to group different types of trials in a search task
        Returns:
            A boolean value representing whether the trial should be stop.
        """
        raise NotImplementedError()

    def __repr__(self):
        return to_repr(self)


def get_percentile_score(history, n_step, group_id, percentile, sign=1):
    trial_scores = get_previous_trials_scores(history, n_step, n_step, group_id)
    percentile_score = np.percentile(trial_scores * sign, percentile) * sign
    return percentile_score


def get_previous_trials_scores(history, from_step, to_step, group_id):
    trial_scores = []
    for trial in history.trials:
        if not trial.succeeded:
            continue
        scores = trial.iteration_scores.get(group_id)
        if scores and len(scores) >= (to_step + 1):
            trial_scores.append(scores[from_step:(to_step + 1)])
    return np.array(trial_scores)
