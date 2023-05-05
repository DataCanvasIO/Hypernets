# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from ..core.searcher import Searcher, OptimizeDirection
from ..core import EarlyStoppingError
from sklearn.model_selection import ParameterGrid


class GridSearcher(Searcher):
    def __init__(self, space_fn, optimize_direction=OptimizeDirection.Minimize, space_sample_validation_fn=None,
                 n_expansion=5):
        Searcher.__init__(self, space_fn, optimize_direction, space_sample_validation_fn=space_sample_validation_fn)
        space = space_fn()
        assignable_params = space.get_unassigned_params()
        self.grid = {}
        self.n_expansion = n_expansion
        for p in assignable_params:
            self.grid[p.id] = [s.value for s in p.expansion(n_expansion)]
        self.all_combinations = list(ParameterGrid(self.grid))
        self.position_ = -1

    @property
    def parallelizable(self):
        return True

    def sample(self, space_options=None):
        sample = self._sample_and_check(self._get_sample)
        return sample

    def _get_sample(self):
        self.position_ += 1

        if self.position_ >= len(self.all_combinations):
            raise EarlyStoppingError('no more samples.')
        sample = self.space_fn()
        for k, v in self.all_combinations[self.position_].items():
            sample.__dict__[k].assign(v)
        assert sample.all_assigned == True
        return sample

    def get_best(self):
        raise NotImplementedError

    def update_result(self, space, result):
        pass

    def reset(self):
        self.position_ = -1

    def export(self):
        raise NotImplementedError


def test_parameter_grid(self):
    space = self.get_space()
    ps = space.get_unassigned_params()
    grid = {}
    for p in ps:
        grid[p.name] = [s.value for s in p.expansion(2)]
    all_vectors = list(ParameterGrid(grid))
    for ps in all_vectors:
        space = self.get_space()
        for k, v in ps.items():
            space.__dict__[k].assign(v)
        assert space.all_assigned == True
