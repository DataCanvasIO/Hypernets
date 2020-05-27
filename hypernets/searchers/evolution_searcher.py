# -*- coding:utf-8 -*-
"""

"""
from ..core.searcher import Searcher, OptimizeDirection


class EvolutionSearcher(Searcher):
    def __init__(self, space_fn, optimize_direction):
        Searcher.__init__(self, space_fn=space_fn, optimize_direction=optimize_direction)

    def sample(self):
        pass

    def update_result(self, space, result):
        pass
