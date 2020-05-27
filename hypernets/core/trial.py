# -*- coding:utf-8 -*-
"""

"""
from ..core.searcher import OptimizeDirection


class Trail():
    def __init__(self, space_sample, trail_no, reward, elapsed):
        self.space_sample = space_sample
        self.trail_no = trail_no
        self.reward = reward
        self.elapsed = elapsed
        pass


class TrailHistory():
    def __init__(self, optimize_direction):
        self.history = []
        self.optimize_direction = optimize_direction

    def append(self, trail):
        old_best = self.get_best()
        self.history.append(trail)
        new_best = self.get_best()
        improved = old_best != new_best
        return improved

    def get_best(self):
        sorted_trials = sorted(self.history,
                               key=lambda
                                   t: t.reward if self.optimize_direction == OptimizeDirection.Minimize else -t.reward)
        if len(sorted_trials) > 0:
            return sorted_trials[0]
        else:
            return None
