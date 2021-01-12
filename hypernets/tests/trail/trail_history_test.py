# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.core.ops import Identity, Choice
from hypernets.core.search_space import HyperSpace, Int, Real
from hypernets.core.trial import TrialHistory, Trial
from .. import test_output_dir


class Test_TrialHistory():
    def test_is_exsited(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                id1 = Identity(p1=Choice(['a', 'b']), p2=Int(1, 100), p3=Real(0, 1.0))
            return space

        th = TrialHistory('min')
        sample = get_space()
        sample.random_sample()
        trial = Trial(sample, 1, 0.99, 100)
        th.append(trial)

        sample2 = get_space()
        sample2.assign_by_vectors(sample.vectors)

        assert th.is_existed(sample2)

        t = th.get_trial(sample2)
        assert t.reward == 0.99
        assert t.elapsed == 100
        assert t.trial_no == 1

        sample3 = get_space()
        sample3.random_sample()

        assert not th.is_existed(sample3)

    def test_save_load(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                id1 = Identity(p1=Choice(['a', 'b']), p2=Int(1, 100), p3=Real(0, 1.0))
            return space

        th = TrialHistory('min')
        sample = get_space()
        sample.assign_by_vectors([0, 1, 0.1])
        trial = Trial(sample, 1, 0.99, 100)
        th.append(trial)

        sample = get_space()
        sample.assign_by_vectors([1, 2, 0.2])
        trial = Trial(sample, 2, 0.9, 50)
        th.append(trial)

        sample = get_space()
        sample.assign_by_vectors([0, 3, 0.3])
        trial = Trial(sample, 3, 0.7, 200)
        th.append(trial)

        filepath = f'{test_output_dir}/history.txt'
        th.save(filepath)

        with open(filepath) as f:
            lines = f.readlines()
            # assert lines == ['min\n', '1|[0, 1, 0.1]|0.99|100\n', '2|[1, 2, 0.2]|0.9|50\n', '3|[0, 3, 0.3]|0.7|200\n']

        history = TrialHistory.load_history(get_space, filepath)
        assert history.optimize_direction == 'min'
        assert len(history.history) == 3
        assert history.history[0].space_sample.vectors == [0, 1, 0.1]
        assert history.history[0].elapsed == 100.0
        assert history.history[0].reward == 0.99
        assert history.history[0].trial_no == 1

        trajectories = history.get_trajectories()
        assert trajectories == ([0.0, 100.0, 150.0, 350.0], [0.0, 0.99, 0.99, 0.99], [0.0, 0.99, 0.9, 0.7], 1, 100.0)
