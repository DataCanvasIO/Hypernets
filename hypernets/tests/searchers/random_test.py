# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import pytest

from hypernets.core.ops import *
from hypernets.core.search_space import *
from hypernets.searchers.random_searcher import RandomSearcher


class Test_RandomSearcher():
    def get_space(self):
        space = HyperSpace()
        with space.as_default():
            p1 = Int(1, 100)
            p2 = Choice(['a', 'b'])
            p3 = Bool()
            p4 = Real(0.0, 1.0)
            id1 = Identity(p1=p1)
            id2 = Identity(p2=p2)(id1)
            id3 = Identity(p3=p3)(id2)
            id4 = Identity(p4=p4)(id3)
        return space

    def test_random_searcher(self):

        searcher = RandomSearcher(self.get_space, space_sample_validation_fn=lambda s: False)
        with pytest.raises(ValueError):
            searcher.sample()

        searcher = RandomSearcher(self.get_space, space_sample_validation_fn=lambda s: True)
        sample = searcher.sample()
        assert sample

        def valid_sample(sample):
            if sample.Param_Bool_1.value:
                return True
            else:
                return False

        searcher = RandomSearcher(self.get_space, space_sample_validation_fn=valid_sample)
        sample = searcher.sample()
        assert sample

    def test_set_random_state(self):
        from hypernets.core import set_random_state
        set_random_state(9527)

        searcher = RandomSearcher(self.get_space)
        vectors = []
        for i in range(1, 10):
            vectors.append(searcher.sample().vectors)
        assert vectors == [[98, 0, 0, 0.96], [9, 0, 0, 0.93], [60, 0, 1, 0.24], [54, 0, 1, 0.7],
                           [25, 0, 1, 0.73], [67, 1, 1, 0.43], [57, 1, 1, 0.05], [49, 0, 0, 0.71], [71, 1, 1, 0.49]]

        set_random_state(None)
        searcher = RandomSearcher(self.get_space)
        vectors = []
        for i in range(1, 10):
            vectors.append(searcher.sample().vectors)
        assert vectors != [[98, 0, 0, 0.96], [9, 0, 0, 0.93], [60, 0, 1, 0.24], [54, 0, 1, 0.7],
                           [25, 0, 1, 0.73], [67, 1, 1, 0.43], [57, 1, 1, 0.05], [49, 0, 0, 0.71], [71, 1, 1, 0.49]]

        set_random_state(9527)
        searcher = RandomSearcher(self.get_space)
        vectors = []
        for i in range(1, 10):
            vectors.append(searcher.sample().vectors)
        assert vectors == [[98, 0, 0, 0.96], [9, 0, 0, 0.93], [60, 0, 1, 0.24], [54, 0, 1, 0.7],
                           [25, 0, 1, 0.73], [67, 1, 1, 0.43], [57, 1, 1, 0.05], [49, 0, 0, 0.71], [71, 1, 1, 0.49]]

        set_random_state(1)
        searcher = RandomSearcher(self.get_space)
        vectors = []
        for i in range(1, 10):
            vectors.append(searcher.sample().vectors)
        assert vectors == [[38, 1, 0, 0.93], [10, 1, 1, 0.15], [17, 1, 0, 0.39], [7, 1, 0, 0.85], [19, 0, 1, 0.44],
                           [29, 1, 0, 0.67], [88, 1, 1, 0.43], [95, 0, 0, 0.8], [10, 1, 1, 0.09]]

        set_random_state(None)
