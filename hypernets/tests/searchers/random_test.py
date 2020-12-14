# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import pytest

from hypernets.core.ops import *
from hypernets.core.search_space import *
from hypernets.searchers.random_searcher import RandomSearcher


class Test_MCTS():
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

