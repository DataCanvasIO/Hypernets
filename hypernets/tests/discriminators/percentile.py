# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.discriminators import PercentileDiscriminator, ProgressivePercentileDiscriminator

from . import history, group_id


class Test_PercentileDiscriminator():
    def test_percentile(self):
        d = PercentileDiscriminator(50, min_trials=5, min_steps=5, stride=1, history=history, optimize_direction='min')
        p1 = d.is_promising([0.9, 0.9, 0.9, 0.9], group_id)
        assert p1 == True

        p2 = d.is_promising([0.9, 0.9, 0.9, 0.9, 0.9], group_id)
        assert p2 == False

        p2 = d.is_promising([0.9, 0.9, 0.9, 0.9, 0.525], group_id)
        assert p2 == False

        p2 = d.is_promising([0.9, 0.9, 0.9, 0.9, 0.524], group_id)
        assert p2 == True

        d = PercentileDiscriminator(0, min_trials=5, min_steps=5, stride=1, history=history, optimize_direction='min')
        p1 = d.is_promising([0.9, 0.9, 0.9, 0.9, 0.50], group_id)
        assert p1 == True
        p1 = d.is_promising([0.9, 0.9, 0.9, 0.9, 0.56], group_id)
        assert p1 == False

        d = PercentileDiscriminator(100, min_trials=5, min_steps=5, stride=1, history=history, optimize_direction='min')
        p1 = d.is_promising([0.9, 0.9, 0.9, 0.9, 0.55], group_id)
        assert p1 == False
        p1 = d.is_promising([0.9, 0.9, 0.9, 0.9, 0.49], group_id)
        assert p1 == True

    def test_progressive_percentile(self):
        d = ProgressivePercentileDiscriminator([100, 90, 80, 60, 50, 40, 30, 20, 10, 0], min_trials=5, min_steps=5,
                                               stride=1,
                                               history=history, optimize_direction='min')
        p1 = d.is_promising([0.1, 0.1, 0.1, 0.1, 0.1], group_id)
        assert p1 == True

        p1 = d.is_promising([0.1, 0.1, 0.1, 0.1, 0.56], group_id)
        assert p1 == False

        p1 = d.is_promising([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], group_id)
        assert p1 == True

        p1 = d.is_promising([0.1, 0.1, 0.1, 0.1, 0.1, 0.45], group_id)
        assert p1 == False

        p1 = d.is_promising([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], group_id)
        assert p1 == True

        p1 = d.is_promising([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.345], group_id)
        assert p1 == False

        p1 = d.is_promising([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], group_id)
        assert p1 == True

        p1 = d.is_promising([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.34], group_id)
        assert p1 == False
