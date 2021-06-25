# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.discriminators import get_previous_trials_scores, get_percentile_score

from . import history, group_id, group_id2


def test_base():
    ts = get_previous_trials_scores(history, 0, 9, group_id)
    assert ts.shape == (5, 10)
    ts = get_previous_trials_scores(history, 0, 8, group_id)
    assert ts.shape == (6, 9)
    ts2 = get_previous_trials_scores(history, 0, 9, group_id2)
    assert ts2.shape == (1, 10)

    def get_0_100_50_percentile_score(n_step, sign=-1):
        s1 = get_percentile_score(history, n_step, group_id, 0, sign)
        s2 = get_percentile_score(history, n_step, group_id, 100, sign)
        s3 = get_percentile_score(history, n_step, group_id, 50, sign)
        return s1, s2, s3

    p1 = get_0_100_50_percentile_score(0)
    assert p1 == (0.9, 0.9, 0.9)

    p2 = get_0_100_50_percentile_score(1)
    assert p2 == (0.85, 0.8, 0.85)

    p3 = get_0_100_50_percentile_score(5)
    assert p3 == (0.45, 0.4, 0.425)

    p4 = get_0_100_50_percentile_score(9)
    assert p4 == (0.25, 0.21, 0.23)

    p5 = get_0_100_50_percentile_score(9, 1)
    assert p5 == (0.21, 0.25, 0.23)
