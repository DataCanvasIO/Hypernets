# -*- coding:utf-8 -*-
"""

"""
from ..core.searcher import Searcher, OptimizeDirection
from ..core import TrialHistory
from ..core.callbacks import EarlyStoppingError


class PlaybackSearcher(Searcher):
    def __init__(self, trail_history: TrialHistory, top_n=20, optimize_direction=OptimizeDirection.Minimize):
        assert trail_history is not None
        assert len(trail_history.trials) > 0
        self.history = trail_history
        self.top_n = top_n
        self.samples = [t.space_sample for t in self.history.get_top(top_n)]
        self.index = 0
        Searcher.__init__(self, None, use_meta_learner=False, optimize_direction=optimize_direction)

    @property
    def parallelizable(self):
        return True

    def sample(self):
        if self.index >= len(self.samples):
            raise EarlyStoppingError('no more samples.')
        sample = self.samples[self.index]
        self.index += 1
        return sample

    def update_result(self, space, result):
        pass
