# -*- coding:utf-8 -*-
"""

"""
from hypernets.core.trial import *
from hypernets.core.search_space import *
from hypernets.core.ops import *


class Test_TrailStore():
    def get_space(self):
        space = HyperSpace()
        with space.as_default():
            id1 = Identity(p1=Choice([1, 2]), p2=Int(1, 100))
            id2 = Identity(p3=Real(0, 1, step=0.2))(id1)
            id3 = Identity(p4=Dynamic(lambda args: args['p5'] * 3, p5=Choice([2, 4, 8])))(id2)
        return space

    def test_basic(self):
        store = DiskTrailStore()
        dataset_id = 'test_dataset'
        sample = self.get_space()
        sample.random_sample()

        trail = Trail(sample, 1, 0.99, 100)
        store.put(dataset_id, trail)
        store.reset()

        trail_get = store.get(dataset_id, sample)
        assert trail.trail_no == 1
        assert trail.reward == 0.99
        assert trail.elapsed == 100
        assert trail.space_sample.vectors == trail_get.space_sample.vectors

        trails = store.get_all(dataset_id, sample.signature)
        assert trails
