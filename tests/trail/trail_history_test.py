# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from .. import test_output_dir
from hypernets.core.trial import TrailHistory, Trail
from hypernets.core.ops import HyperSpace, Identity, Choice, Int, Real


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

    def test_save_load(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                id1 = Identity(p1=Choice(['a', 'b']), p2=Int(1, 100), p3=Real(0, 1.0))
            return space

        th = TrailHistory('min')
        sample = get_space()
        sample.assign_by_vectors([0, 1, 0.1])
        trail = Trail(sample, 1, 0.99, 100)
        th.append(trail)

        sample = get_space()
        sample.assign_by_vectors([1, 2, 0.2])
        trail = Trail(sample, 2, 0.9, 50)
        th.append(trail)

        sample = get_space()
        sample.assign_by_vectors([0, 3, 0.3])
        trail = Trail(sample, 3, 0.7, 200)
        th.append(trail)

        filepath = f'{test_output_dir}/history.txt'
        th.save(filepath)

        with open(filepath) as f:
            lines = f.readlines()
            assert lines == ['1|[0, 1, 0.1]|0.99|100\n', '2|[1, 2, 0.2]|0.9|50\n', '3|[0, 3, 0.3]|0.7|200\n']

        history = TrailHistory.load_history(get_space, filepath)
        assert len(history) == 3
        assert history[0].space_sample.vectors == [0, 1, 0.1]
        assert history[0].elapsed == 100.0
        assert history[0].reward == 0.99
        assert history[0].trail_no == 1
