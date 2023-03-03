import abc
from typing import List

import numpy as np

from hypernets.core import Searcher, OptimizeDirection
from hypernets.core.objective import Objective
from hypernets.searchers.genetic import Individual


def dominate(x1: np.ndarray, x2: np.ndarray, directions=None):
    # return: is s1 dominate s2
    if directions is None:
        directions = ['min'] * x1.shape[0]

    ret = []
    for i, j in enumerate(range(x1.shape[0])):
        if directions[i] == 'min':
            if x1[j] < x2[j]:
                ret.append(1)
            elif x1[j] == x2[j]:
                ret.append(0)
            else:
                return False  # s1 does not dominate s2
        else:
            if x1[j] > x2[j]:
                ret.append(1)
            elif x1[j] == x2[j]:
                ret.append(0)
            else:
                return False
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


class MOOSearcher(Searcher, metaclass=abc.ABCMeta):

    def __init__(self, space_fn, objectives: List[Objective], *, use_meta_learner=True,
                 space_sample_validation_fn=None, **kwargs):
        super().__init__(space_fn=space_fn, use_meta_learner=use_meta_learner,
                         space_sample_validation_fn=space_sample_validation_fn, **kwargs)
        self.objectives = objectives

    @abc.abstractmethod
    def get_nondominated_set(self) -> List[Individual]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_historical_population(self) -> List[Individual]:
        raise NotImplementedError

    def plot_pf(self, consistent_direction=False):
        def do_P(indis, color, label, fig):
            indis_array = np.array(list(map(lambda _: _.scores, indis)))
            fig.scatter(indis_array[:, 0], indis_array[:, 1], c=color, label=label)

        try:
            from matplotlib import pyplot as plt
        except Exception:
            raise RuntimeError("it requires matplotlib installed.")

        if len(self.objectives) != 2:
            raise RuntimeError("plot currently works only in case of 2 objectives. ")

        objective_names = list(map(lambda v: v.name, self.objectives))
        population = self.get_historical_population()
        if consistent_direction:
            scores_array = np.array([indi.scores for indi in population])
            reverse_inx = []
            if len(set(map(lambda v: v.direction, self.objectives))) > 1:
                for i, o in enumerate(self.objectives):
                    if o.direction != 'min':
                        objective_names[i] = f"{objective_names[i]}(e^-x)"
                        reverse_inx.append(i)

            reversed_scores = scores_array.copy()
            reversed_scores[:, reverse_inx] = np.exp(-scores_array[:, reverse_inx])  # e^-x

            rd_population = [Individual(indi.dna, reversed_scores[i], indi.random_state)
                             for i, indi in enumerate(population)]
            fixed_population = rd_population
        else:
            fixed_population = population

        figure = plt.figure(figsize=(6, 6))
        def do_P(indis, color, label, fig):
            indis_array = np.array(list(map(lambda _: _.scores, indis)))
            plt.scatter(indis_array[:, 0], indis_array[:, 1], c=color, label=label)

        ns: List[Individual] = calc_nondominated_set(fixed_population)
        do_P(ns, color='red', label='nondominated', fig=figure)

        ds: List[Individual] = list(filter(lambda v: v not in ns, fixed_population))
        do_P(ds, color='blue', label='dominated', fig=figure)

        figure.legend()
        plt.xlabel(objective_names[0])
        plt.ylabel(objective_names[1])

        return figure
