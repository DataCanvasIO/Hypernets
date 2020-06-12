# -*- coding:utf-8 -*-
"""

"""
from ..core.searcher import Searcher, OptimizeDirection


class RandomSearcher(Searcher):
    def __init__(self, space_fn, optimize_direction=OptimizeDirection.Minimize, dataset_id=None, trail_store=None):
        Searcher.__init__(self, space_fn, optimize_direction, dataset_id=dataset_id, trail_store=trail_store)

    def sample(self, history):
        space = self.space_fn()
        space.random_sample()
        return space

    def get_best(self):
        raise NotImplementedError

    def update_result(self, space, result):
        pass

    def reset(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
