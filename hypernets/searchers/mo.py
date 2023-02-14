from typing import List

import numpy as np

from hypernets.searchers.genetic import Individual


def dominate(x1: np.ndarray, x2: np.ndarray):
    # return: is s1 dominate s2
    ret = []
    for j in range(x1.shape[0]):
        if x1[j] < x2[j]:
            ret.append(1)
        elif x1[j] == x2[j]:
            ret.append(0)
        else:
            return False  # s1 does not dominate s2
    if np.sum(np.array(ret)) >= 1:
        return True  # s1 has at least one metric better that s2
    else:
        return False


def _compair(x1, x2, c_op):

    x1 = np.array(x1)
    x2 = np.array(x2)

    ret = []
    for j in range(x1.shape[0]):
        if c_op(x1[j], x2[j]):
            ret.append(1)
        elif np.equal([j], x2[j]):
            ret.append(0)
        else:
            return False  # x1 does not dominate x2

    if np.sum(np.array(ret)) >= 1:
        return True  # x1 has at least one metric better that x2
    else:
        return False


def calc_nondominated_set(population: List[Individual]):
    def find_non_dominated_solu(indi):
        if (np.array(indi.scores) == None).any():  # illegal individual for the None scores
            return False
        for indi_ in population:
            if indi_ == indi:
                continue
            if dominate(indi_.scores, indi.scores):
                return False
        return True  # this is a pareto optimal

    # find non-dominated solution for every solution
    nondominated_set = list(filter(lambda s: find_non_dominated_solu(s), population))

    return nondominated_set


def _op_less(self, x1, x2):
    return self._compair(x1, x2, np.less)


def _op_greater(self, x1, x2):
    return self._compair(x1, x2, np.greater)
