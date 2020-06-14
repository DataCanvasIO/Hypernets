# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.core.trial import *
from hypernets.core.search_space import *
from hypernets.core.ops import *


class Test_TrialHistory():
    def test_is_exsited(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                id1 = Identity(p1=Choice(['a', 'b']), p2=Int(1, 100), p3=Real(0, 1.0))
            return space

        th = TrailHistory('min')
        sample = get_space()
        sample.random_sample()
        trail = Trail(sample, 1, 0.99, 100)
        th.append(trail)

        sample2 = get_space()
        sample2.assign_by_vectors(sample.vectors)

        assert th.is_existed(sample2)

        t = th.get_trail(sample2)
        assert t.reward == 0.99
        assert t.elapsed == 100
        assert t.trail_no == 1

        sample3 = get_space()
        sample3.random_sample()

        assert not th.is_existed(sample3)
