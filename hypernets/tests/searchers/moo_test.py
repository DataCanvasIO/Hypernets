import numpy as np

from hypernets.searchers.moo import pareto_dominate, calc_nondominated_set
from hypernets.searchers.genetic import Individual


def test_dominate():
    s1 = np.array([0.5, 0.6])
    s2 = np.array([0.4, 0.6])
    assert pareto_dominate(s2, s1) is True

    s3 = np.array([0.3, 0.7])
    assert pareto_dominate(s2, s3) is False

    s4 = np.array([0.2, 0.5])
    assert pareto_dominate(s3, s4) is False

    # different direction
    s5 = np.array([0.8, 100])
    s6 = np.array([0.7, 101])
    assert pareto_dominate(s5, s6, directions=('max', 'min')) is True


def test_calc_nondominated_set():
    i1 = Individual("1", np.array([0.1, 0.2]), None)
    i2 = Individual("1", np.array([0.2, 0.1]), None)
    i3 = Individual("1", np.array([0.2, 0.2]), None)
    i4 = Individual("1", np.array([0.3, 0.2]), None)
    i5 = Individual("1", np.array([0.4, 0.4]), None)
    nondominated_set = calc_nondominated_set([i1, i2, i3, i4, i5])
    assert len(nondominated_set) == 2
    assert i1 in nondominated_set
    assert i2 in nondominated_set
