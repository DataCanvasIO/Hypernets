# -*- coding:utf-8 -*-
"""

"""
from .stateful import Stateful
import enum


class OptimizeDirection(enum.Enum):
    Minimize = 'min'
    Maximize = 'max'


class Searcher(Stateful):
    def __init__(self, space_fn, optimize_direction=OptimizeDirection.Minimize, use_meta_learner=True,
                 space_sample_validation_fn=None):
        self.space_fn = space_fn
        self.use_meta_learner = use_meta_learner
        self.optimize_direction = optimize_direction
        self.meta_learner = None
        self.space_sample_validation_fn = space_sample_validation_fn

    def set_meta_learner(self, meta_learner):
        self.meta_learner = meta_learner

    @property
    def parallelizable(self):
        return False

    def sample(self):
        raise NotImplementedError

    def _random_sample(self):
        space_sample = self.space_fn()
        space_sample.random_sample()
        return space_sample

    def _sample_and_check(self, sample_fn):
        counter = 0
        while True:
            space_sample = sample_fn()
            counter += 1
            if counter >= 1000:
                raise ValueError('Unable to take valid sample and exceed the retry limit 1000.')
            if self.space_sample_validation_fn is not None:
                if self.space_sample_validation_fn(space_sample):
                    break
            else:
                break
        return space_sample

    def get_best(self):
        raise NotImplementedError

    def update_result(self, space, result):
        raise NotImplementedError

    def summary(self):
        return 'No Summary'

    def reset(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
