import abc
from typing import List
from operator import attrgetter

import numpy as np

from hypernets.core import Searcher, OptimizeDirection, pareto
from hypernets.core.objective import Objective
from hypernets.searchers.genetic import Individual
from hypernets.utils import const


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

    def get_pareto_nondominated_set(self):
        population = self.get_historical_population()
        scores = np.array([_.scores for _ in population])
        obj_directions = [_.direction for _ in self.objectives]
        non_dominated_inx = pareto.calc_nondominated_set(scores, directions=obj_directions)
        return [population[i] for i in non_dominated_inx]

    def _do_plot(self, indis, color, label, ax, marker):
        if len(indis) <= 0:
            return
        indis_scores = np.asarray(list(map(attrgetter("scores"), indis)))
        ax.scatter(indis_scores[:, 0], indis_scores[:, 1], c=color, label=label,  marker=marker)

    def _plot_pareto(self, ax, historical_individuals):
        # pareto dominated plot
        pn_set = self.get_pareto_nondominated_set()
        pd_set: List[Individual] = list(filter(lambda v: v not in pn_set, historical_individuals))
        self._do_plot(pn_set, color='red', label='non-dominated', ax=ax, marker="o")  # , marker="o"
        self._do_plot(pd_set, color='blue', label='dominated', ax=ax, marker="o")
        ax.set_title(f"non-dominated solution (total={len(historical_individuals)}) in pareto scene")
        objective_names = [_.name for _ in self.objectives]
        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.legend()

    def _sub_plot_pop(self, ax, historical_individuals):
        population = self.get_population()
        not_in_population: List[Individual] = list(filter(lambda v: v not in population, historical_individuals))
        self._do_plot(population, color='red', label='in-population', ax=ax, marker="o")  #
        self._do_plot(not_in_population, color='blue', label='others', ax=ax, marker="o")  # marker="p"
        ax.set_title(f"individual in population(total={len(historical_individuals)}) plot")
        # handles, labels = ax.get_legend_handles_labels()
        objective_names = [_.name for _ in self.objectives]
        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.legend()

    @abc.abstractmethod
    def get_historical_population(self) -> List[Individual]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_population(self) -> List[Individual]:
        raise NotImplementedError

    def _plot_population(self, figsize, **kwargs):
        raise NotImplementedError

    def check_plot(self):
        try:
            from matplotlib import pyplot as plt
        except Exception:
            raise RuntimeError("it requires matplotlib installed.")

        if len(self.objectives) != 2:
            raise RuntimeError("plot currently works only in case of 2 objectives. ")

    def plot_population(self, figsize=(6, 6), **kwargs):
        self.check_plot()
        figs, axes = self._plot_population(figsize=figsize, **kwargs)
        return figs, axes

    def kind(self):
        return const.SEARCHER_MOO
