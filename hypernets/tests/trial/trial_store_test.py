# -*- coding:utf-8 -*-
"""

"""
from hypernets.core.trial import *
from hypernets.core.search_space import *
from hypernets.core.ops import *
from hypernets.tests import test_output_dir


class Test_TrialStore():
    def get_space(self):
        space = HyperSpace()
        with space.as_default():
            id1 = Identity(p1=Choice([1, 2]), p2=Int(1, 100))
            id2 = Identity(p3=Real(0, 1, step=0.2))(id1)
            id3 = Identity(p4=Dynamic(lambda p5: p5 * 3, p5=Choice([2, 4, 8])))(id2)
        return space

    def test_basic(self):
        store = DiskTrialStore(f'{test_output_dir}/trial_store')
        dataset_id = 'test_dataset'
        sample = self.get_space()
        sample.random_sample()

        trial = Trial(sample, 1, 0.99, 100)
        store.put(dataset_id, trial)
        store.reset()

        trial_get = store.get(dataset_id, sample)
        assert trial.trial_no == 1
        assert trial.reward == 0.99
        assert trial.elapsed == 100
        assert trial.space_sample.vectors == trial_get.space_sample.vectors

        trials = store.get_all(dataset_id, sample.signature)
        assert trials
