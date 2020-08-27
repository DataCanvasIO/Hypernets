# -*- coding:utf-8 -*-
"""

"""
from ..core.searcher import Searcher, OptimizeDirection


class RandomSearcher(Searcher):
    def __init__(self, space_fn, optimize_direction=OptimizeDirection.Minimize, space_sample_validation_fn=None):
        Searcher.__init__(self, space_fn, optimize_direction, space_sample_validation_fn=space_sample_validation_fn)

    @property
    def parallelizable(self):
        return True

    def sample(self):
        sample = self._sample_and_check(self._random_sample)
        return sample

    def get_best(self):
        raise NotImplementedError

    def update_result(self, space, result):
        pass

    def reset(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
