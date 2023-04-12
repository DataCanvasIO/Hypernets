import numpy as np

from hypernets.core.pareto import pareto_dominate
from hypernets.searchers.genetic import Individual


def test_dominate():
    s1 = np.array([0.5, 0.6])
    s2 = np.array([0.4, 0.6])
    assert pareto_dominate(s2, s1)

    s3 = np.array([0.3, 0.7])
    assert not pareto_dominate(s2, s3)

    s4 = np.array([0.2, 0.5])
    assert not pareto_dominate(s3, s4)

    # different direction
    s5 = np.array([0.8, 100])
    s6 = np.array([0.7, 101])
    assert pareto_dominate(s5, s6, directions=('max', 'min'))
