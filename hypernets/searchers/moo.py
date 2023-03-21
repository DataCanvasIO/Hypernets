import abc
from typing import List
from operator import attrgetter

import numpy as np

from hypernets.core import Searcher, OptimizeDirection
from hypernets.core.objective import Objective
from hypernets.searchers.genetic import Individual, Survival
from hypernets.utils import const


def pareto_dominate(x1: np.ndarray, x2: np.ndarray, directions=None):
    #  dominance in pareto scene
    if directions is None:
        directions = ['min'] * x1.shape[0]

    ret = []
    for i in range(x1.shape[0]):
        if directions[i] == 'min':
            if x1[i] < x2[i]:
                ret.append(1)
            elif x1[i] == x2[i]:
                ret.append(0)
            else:
                return False  # s1 does not dominate s2
        else:
            if x1[i] > x2[i]:
                ret.append(1)
            elif x1[i] == x2[i]:
                ret.append(0)
            else:
                return False

    # s1 has at least one metric better that s2
    return np.sum(np.array(ret)) >= 1


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

    @abc.abstractmethod
    def get_population(self) -> List[Individual]:
        raise NotImplementedError

    def _plot(self, pop: List[Individual], label: str, comparison_label: str, **kwargs):
        def do_P(indis, color, label, ax, marker):
            if len(indis) <= 0:
                return
            indis_array = np.array(list(map(attrgetter("scores"), indis)))
            ax.scatter(indis_array[:, 0], indis_array[:, 1], c=color, label=label,  marker=marker)

        try:
            from matplotlib import pyplot as plt
        except Exception:
            raise RuntimeError("it requires matplotlib installed.")

        if len(self.objectives) != 2:
            raise RuntimeError("plot currently works only in case of 2 objectives. ")

        objective_names = list(map(lambda v: v.name, self.objectives))
        historical_individuals = self.get_historical_population()
        comparison: List[Individual] = list(filter(lambda v: v not in pop, historical_individuals))

        fig, ax = plt.subplots(figsize=(10, 10))

        do_P(pop, color='red', label=label, ax=ax, marker="o")

        do_P(comparison, color='blue', marker="p", label=comparison_label, ax=ax)
        # ax.set_ylim(-1, 0)
        # ax.set_xlim(-1, 0)

        ax, fig = self.plot_addition(ax, fig, **kwargs)

        fig.legend(loc='upper right')
        plt.xlabel(objective_names[0])
        plt.ylabel(objective_names[1])
        # plt.show()
        return fig

    def plot_addition(self, ax, fig, **kwargs):
        return ax, fig

    def plot_nondominated(self, **kwargs):
        ns: List[Individual] = self.get_nondominated_set()
        self._plot(ns, label='nondominated', comparison_label='dominated', **kwargs)

    def plot_population(self, **kwargs):
        pop: List[Individual] = self.get_population()
        self._plot(pop, label='in population', comparison_label='the others', **kwargs)

    def kind(self):
        return const.SEARCHER_MOO

    # def plot_pf(self, consistent_direction=False):
    #     def do_P(indis, color, label, fig):
    #         indis_array = np.array(list(map(lambda _: _.scores, indis)))
    #         fig.scatter(indis_array[:, 0], indis_array[:, 1], c=color, label=label)
    #
    #     try:
    #         from matplotlib import pyplot as plt
    #     except Exception:
    #         raise RuntimeError("it requires matplotlib installed.")
    #
    #     if len(self.objectives) != 2:
    #         raise RuntimeError("plot currently works only in case of 2 objectives. ")
    #
    #     objective_names = list(map(lambda v: v.name, self.objectives))
    #     population = self.get_historical_population()
    #     if consistent_direction:
    #         scores_array = np.array([indi.scores for indi in population])
    #         reverse_inx = []
    #         if len(set(map(lambda v: v.direction, self.objectives))) > 1:
    #             for i, o in enumerate(self.objectives):
    #                 if o.direction != 'min':
    #                     objective_names[i] = f"{objective_names[i]}(e^-x)"
    #                     reverse_inx.append(i)
    #
    #         reversed_scores = scores_array.copy()
    #         reversed_scores[:, reverse_inx] = np.exp(-scores_array[:, reverse_inx])  # e^-x
    #
    #         rd_population = [Individual(indi.dna, reversed_scores[i], indi.random_state)
    #                          for i, indi in enumerate(population)]
    #         fixed_population = rd_population
    #     else:
    #         fixed_population = population
    #
    #     figure = plt.figure(figsize=(6, 6))
    #     def do_P(indis, color, label, fig):
    #         indis_array = np.array(list(map(lambda _: _.scores, indis)))
    #         plt.scatter(indis_array[:, 0], indis_array[:, 1], c=color, label=label)
    #
    #     ns: List[Individual] = calc_nondominated_set(fixed_population)
    #     do_P(ns, color='red', label='nondominated', fig=figure)
    #
    #     ds: List[Individual] = list(filter(lambda v: v not in ns, fixed_population))
    #     do_P(ds, color='blue', label='dominated', fig=figure)
    #
    #     figure.legend()
    #     plt.xlabel(objective_names[0])
    #     plt.ylabel(objective_names[1])
    #
    #     return figure
