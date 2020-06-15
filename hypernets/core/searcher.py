# -*- coding:utf-8 -*-
"""

"""
from .stateful import Stateful
import enum


class OptimizeDirection(enum.Enum):
    Minimize = 'min'
    Maximize = 'max'


class Searcher(Stateful):
    def __init__(self, space_fn, optimize_direction=OptimizeDirection.Minimize, use_meta_learner=True):
        self.space_fn = space_fn
        self.use_meta_learner = use_meta_learner
        self.optimize_direction = optimize_direction
        self.meta_learner = None

    def set_meta_learner(self, meta_learner):
        self.meta_learner = meta_learner

    def sample(self):
        raise NotImplementedError

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
