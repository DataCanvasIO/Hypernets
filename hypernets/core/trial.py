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
        top1 = self.get_top(1)
        if len(top1) <= 0:
            return None
        else:
            return top1[0]

    def get_top(self, n=10):
        if len(self.history) <= 0:
            return []
        sorted_trials = sorted(self.history,
                               key=lambda
                                   t: t.reward if self.optimize_direction == OptimizeDirection.Minimize else -t.reward)
        if n > len(sorted_trials):
            n = len(sorted_trials)
        return sorted_trials[:n]

    def get_space_signatures(self):
        signatures = set()
        for s in [t.space_sample for t in self.history]:
            signatures.add(s.signature)
        return signatures

    def diff(self, trails):
        signatures = set()
        for s in [t.space_sample for t in trails]:
            signatures.add(s.signature)

        diffs = {}
        for sign in signatures:
            ts = [t for t in trails if t.space_sample.signature == sign]
            pv_dict = {}
            for t in ts:
                for p in t.space_sample.get_assignable_params():
                    k = p.alias
                    v = str(p.value)
                    if pv_dict.get(k) is None:
                        pv_dict[k] = {}
                    if pv_dict[k].get(v) is None:
                        pv_dict[k][v] = [t.reward]
                    else:
                        pv_dict[k][v].append(t.reward)
            diffs[sign] = pv_dict

        return diffs
