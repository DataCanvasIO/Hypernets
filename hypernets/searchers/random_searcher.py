# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from ..core.searcher import Searcher, OptimizeDirection


class RandomSearcher(Searcher):
    def __init__(self, space_fn, optimize_direction=OptimizeDirection.Minimize):
        Searcher.__init__(self, space_fn, optimize_direction)

    def sample(self):
        space = self.space_fn()
        space.random_sample()
        space.compile_space()
        return space

    def get_best(self):
        raise NotImplementedError

    def update_result(self, space_id, params, result):
        pass

    def reset(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
