# -*- coding:utf-8 -*-
"""

"""
from .stateful import Stateful
import enum


class OptimizeDirection(enum.Enum):
    Minimize = 'min'
    Maximize = 'max'


class Searcher(Stateful):
    def __init__(self, space_fn, optimize_direction=OptimizeDirection.Minimize, dataset_id=None, trail_store=None):
        self.space_fn = space_fn
        self.trail_store = trail_store
        self.dataset_id = dataset_id
        self.optimize_direction = optimize_direction

    def sample(self, history):
        raise NotImplementedError

    def get_best(self):
        raise NotImplementedError

    def update_result(self, space, result):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
