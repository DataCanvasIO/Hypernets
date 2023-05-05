# -*- coding:utf-8 -*-
"""

"""
from ..core import TrialHistory
from ..core.callbacks import EarlyStoppingError
from ..core.searcher import Searcher, OptimizeDirection


class PlaybackSearcher(Searcher):
    def __init__(self, history: TrialHistory, top_n=None, reverse=False,
                 optimize_direction=OptimizeDirection.Minimize):
        assert history is not None
        assert len(history.trials) > 0

        self.history = history
        self.top_n = top_n if top_n is not None else len(history.trials)
        self.samples = [t.space_sample for t in self.history.get_top(self.top_n)]
        self.index = 0
        self.reverse = reverse

        if reverse:
            self.samples.reverse()

        super(PlaybackSearcher, self).__init__(None, use_meta_learner=False, optimize_direction=optimize_direction)

    @property
    def parallelizable(self):
        return True

    def sample(self, space_options=None):
        if self.index >= len(self.samples):
            raise EarlyStoppingError('no more samples.')
        sample = self.samples[self.index]
        self.index += 1
        return sample

    def update_result(self, space, result):
        pass
