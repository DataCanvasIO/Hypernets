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


def calc_nondominated_set(solutions: List[Individual]):
    def find_non_dominated_solu(input_solu):
        for solu in solutions:
            if solu == input_solu:
                continue
            if dominate(solu.scores, input_solu.scores):
                # solu_i has non-dominated solution
                return solu
        return None  # this is a pareto optimal

    # find non-dominated solution for every solution

    solutions_filtered = list(filter(lambda s: (np.array(s.scores) != None).all(), solutions))

    nondominated_set = list(filter(lambda s: find_non_dominated_solu(s) is None, solutions_filtered))

    return nondominated_set


def _op_less(self, x1, x2):
    return self._compair(x1, x2, np.less)


def _op_greater(self, x1, x2):
    return self._compair(x1, x2, np.greater)
