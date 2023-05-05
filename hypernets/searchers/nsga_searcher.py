from typing import List
from functools import cmp_to_key

import numpy as np

from hypernets.utils import logging as hyn_logging, const
from ..core.pareto import pareto_dominate
from ..core import HyperSpace, get_random_state

from .moo import MOOSearcher
from .genetic import Individual, SinglePointMutation, _Survival, create_recombination

logger = hyn_logging.get_logger(__name__)


class _NSGAIndividual(Individual):
    def __init__(self, dna: HyperSpace, scores: np.ndarray, random_state):

        super().__init__(dna, scores, random_state)

        self.dna = dna
        self.scores = scores

        self.rank: int = -1  # rank starts from 1

        self.S: List[_NSGAIndividual] = []
        self.n: int = -1

        self.T: List[_NSGAIndividual] = []

        self.distance: float = -1.0  # crowding-distance

    def reset(self):
        self.rank = -1
        self.S = []
        self.n = 0
        self.T = []
        self.distance = -1.0

    def __repr__(self):
        return f"{self.__class__.__name__}(scores={self.scores}, " \
               f"rank={self.rank}, n={self.n}, distance={self.distance})"


class _RankAndCrowdSortSurvival(_Survival):

    def __init__(self, directions, population_size, random_state):
        self.directions = directions
        self.population_size = population_size
        self.random_state = random_state

    @staticmethod
    def crowding_distance_assignment(I: List[_NSGAIndividual]):
        scores_array = np.array([indi.scores for indi in I])

        maximum_array = np.max(scores_array, axis=0)
        minimum_array = np.min(scores_array, axis=0)

        for m in range(len(I[0].scores)):
            sorted_I = list(sorted(I, key=lambda v: v.scores[m], reverse=False))
            sorted_I[0].distance = float("inf")  # so that boundary points always selected, because they are not crowd
            sorted_I[len(I) - 1].distance = float("inf")
            # only assign distances for non-boundary points
            for i in range(len(I) - 2):
                ti = i + 1
                sorted_I[ti].distance = sorted_I[ti].distance \
                                        + (sorted_I[ti + 1].scores[m] - sorted_I[ti - 1].scores[m]) \
                                        / (maximum_array[m] - minimum_array[m])
        return I

    def fast_non_dominated_sort(self, pop: List[_NSGAIndividual]):
        for p in pop:
            p.reset()
        F_1 = []
        F = [F_1]  # to store pareto front of levels respectively
        for p in pop:
            p.n = 0
            for q in pop:
                if p == q:
                    continue
                if self.dominate(p, q, pop=pop):
                    p.S.append(q)
                if self.dominate(q, p, pop=pop):
                    p.T.append(q)
                    p.n = p.n + 1

            if p.n == 0:
                p.rank = 0
                F_1.append(p)

        i = 0
        while True:
            Q = []
            for p in F[i]:
                for q in p.S:
                    q.n = q.n - 1
                    if q.n == 0:
                        q.rank = i + 1
                        Q.append(q)
            if len(Q) == 0:
                break
            F.append(Q)
            i = i + 1
        return F

    def sort_font(self, front: List[_NSGAIndividual]):
        return self.crowding_distance_assignment(front)

    def sort_population(self, population: List[_NSGAIndividual]):
        return sorted(population, key=self.cmp_operator, reverse=False)

    def update(self, pop: List[_NSGAIndividual], challengers: List[Individual]):
        temp_pop = []
        temp_pop.extend(pop)
        temp_pop.extend(challengers)
        if len(pop) < self.population_size:
            return temp_pop

        # assign a weighted Euclidean distance for each one
        p_sorted = self.fast_non_dominated_sort(temp_pop)
        if len(p_sorted) == 1 and len(p_sorted[0]) == 0:
            print(f"ERR: {p_sorted}")

        # sort individual in a front
        p_selected: List[_NSGAIndividual] = []
        for rank, P_front in enumerate(p_sorted):
            if len(P_front) == 0:
                break
            individuals = self.sort_font(P_front)  # only assign distance for nsga
            p_selected.extend(individuals)
            if len(p_selected) >= self.population_size:
                break

        # ensure population size
        p_cmp_sorted = list(sorted(p_selected, key=cmp_to_key(self.cmp_operator), reverse=True))
        p_final = p_cmp_sorted[:self.population_size]
        logger.debug(f"Individual {p_cmp_sorted[self.population_size-1: ]} have been removed from population,"
                     f" sorted population ={p_cmp_sorted}")

        return p_final

    def dominate(self, ind1: _NSGAIndividual, ind2: _NSGAIndividual, pop: List[_NSGAIndividual]):
        return pareto_dominate(x1=ind1.scores, x2=ind2.scores, directions=self.directions)

    @staticmethod
    def cmp_operator(s1: _NSGAIndividual, s2: _NSGAIndividual):
        if s1.rank < s2.rank:
            return 1
        elif s1.rank == s2.rank:
            if s1.distance > s2.distance:  # the larger the distance the better
                return 1
            elif s1.distance == s2.distance:
                return 0
            else:
                return -1
        else:
            return -1

    def calc_nondominated_set(self, population: List[_NSGAIndividual]):
        def find_non_dominated_solu(indi):
            if (np.array(indi.scores) == None).any():  # illegal individual for the None scores
                return False
            for indi_ in population:
                if indi_ == indi:
                    continue
                if self.dominate(ind1=indi_, ind2=indi, pop=population):
                    return False
            return True  # this is a pareto optimal

        # find non-dominated solution for every solution
        ns = list(filter(lambda s: find_non_dominated_solu(s), population))

        return ns


class _NSGAIIBasedSearcher(MOOSearcher):
    def __init__(self, space_fn, objectives, survival, recombination, mutate_probability,
                 space_sample_validation_fn, random_state):

        super().__init__(space_fn=space_fn, objectives=objectives, use_meta_learner=False,
                         space_sample_validation_fn=space_sample_validation_fn)

        self.population: List[_NSGAIndividual] = []
        self.random_state = random_state if random_state is not None else get_random_state()

        if recombination is None:
            self.recombination = create_recombination(const.COMBINATION_SINGLE_POINT, random_state=self.random_state)
        else:
            self.recombination = recombination

        self.mutation = SinglePointMutation(self.random_state, mutate_probability)

        self.survival = survival

        self._historical_individuals: List[_NSGAIndividual] = []

    def binary_tournament_select(self, population):
        indi_inx = self.random_state.randint(low=0, high=len(population) - 1, size=2)  # fixme: maybe duplicated inx

        p1 = population[indi_inx[0]]
        p2 = population[indi_inx[1]]

        # select the first parent
        if self.survival.cmp_operator(p1, p2) >= 0:
            first_inx = indi_inx[0]
        else:
            first_inx = indi_inx[1]

        # select the second parent
        indi_inx = self.random_state.randint(low=0, high=len(population) - 1, size=2)
        try_times = 0
        while first_inx in indi_inx:  # exclude the first individual
            if try_times > 10000:
                raise RuntimeError("too many times for selecting individual to mat. ")
            indi_inx = self.random_state.randint(low=0, high=len(population) - 1, size=2)
            try_times = try_times + 1

        if self.survival.cmp_operator(p1, p2) >= 0:
            second_inx = indi_inx[0]
        else:
            second_inx = indi_inx[1]

        return population[first_inx], population[second_inx]

    @property
    def directions(self):
        return [o.direction for o in self.objectives]

    def sample(self, space_options=None):
        if space_options is None:
            space_options = {}

        if len(self.population) < self.survival.population_size:
            return self._sample_and_check(self._random_sample, space_options=space_options)

        # binary tournament selection operation
        p1, p2 = self.binary_tournament_select(self.population)

        if self.recombination.check_parents(p1, p2):
            offspring = self.recombination.do(p1, p2, self.space_fn(**space_options))
            final_offspring = self.mutation.do(offspring, self.space_fn(**space_options))
        else:
            final_offspring = self.mutation.do(p1.dna, self.space_fn(**space_options), proba=1)

        return final_offspring

    def get_best(self):
        return list(map(lambda v: v.dna, self.get_nondominated_set()))

    def update_result(self, space, result):
        indi = _NSGAIndividual(space, result, self.random_state)
        self._historical_individuals.append(indi)  # add to history
        p = self.survival.update(pop=self.population,  challengers=[indi])
        self.population = p

        challengers = [indi]

        if challengers[0] in self.population:
            logger.debug(f"new individual{challengers} is accepted by population, current population {self.population}")
        else:
            logger.debug(f"new individual{challengers[0]} is not accepted by population, "
                         f"current population {self.population}")

    def get_nondominated_set(self):
        population = self.get_historical_population()
        ns = self.survival.calc_nondominated_set(population)
        return ns

    def get_historical_population(self):
        return self._historical_individuals

    def get_population(self) -> List[Individual]:
        return self.population

    def _sub_plot_ranking(self, ax, historical_individuals):
        p_sorted = self.survival.fast_non_dominated_sort(historical_individuals)
        colors = ['c', 'm', 'y', 'r', 'g']
        n_colors = len(colors)
        for i, front in enumerate(p_sorted[: n_colors]):
            scores = np.array([_.scores for _ in front])
            ax.scatter(scores[:, 0], scores[:, 1], color=colors[i], label=f"rank={i + 1}")

        if len(p_sorted) > n_colors:
            others = []
            for front in p_sorted[n_colors:]:
                others.extend(front)
            scores = np.array([_.scores for _ in others])
            ax.scatter(scores[:, 0], scores[:, 1], color='b', label='others')
        ax.set_title(f"individuals(total={len(historical_individuals)}) ranking plot")
        objective_names = [_.name for _ in self.objectives]
        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.legend()

    def _plot_population(self, figsize=(6, 6), **kwargs):
        from matplotlib import pyplot as plt

        figs, axes = plt.subplots(3, 1, figsize=(figsize[0], figsize[0] * 3))
        historical_individuals = self.get_historical_population()

        # 1. ranking plot
        self._sub_plot_ranking(axes[0], historical_individuals)

        # 2. population plot
        self._sub_plot_pop(axes[1], historical_individuals)

        # 3. dominated plot
        self._plot_pareto(axes[2], historical_individuals)

        return figs, axes

    def reset(self):
        pass

    def export(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(objectives={self.objectives}, " \
               f"recombination={self.recombination}), " \
               f"mutation={self.mutation}), " \
               f"survival={self.survival}), " \
               f"random_state={self.random_state}"


class NSGAIISearcher(_NSGAIIBasedSearcher):
    """An implementation of "NSGA-II".

    Parameters
    ----------
    space_fn: callable, required
        A search space function which when called returns a `HyperSpace` instance

    objectives: List[Objective], optional, (default to NumOfFeatures instance)
        The optimization objectives.

    recombination: Recombination, required
        the strategy to recombine DNA of parents to generate offspring. Builtin strategies:
        - ShuffleCrossOver
        - UniformCrossover
        - SinglePointCrossOver

    mutate_probability: float, optional, default to 0.7
        the probability of genetic variation for offspring, when the parents can not recombine,
        it will definitely mutate a gene for the generated offspring.

    population_size: int, default to 30
        size of population

    space_sample_validation_fn: callable or None, (default=None)
        used to verify the validity of samples from the search space, and can be used to add specific constraint
        rules to the search space to reduce the size of the space.

    random_state: np.RandomState, optional
        used to reproduce the search process


    References
    ----------

        [1] K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, April 2002, doi: 10.1109/4235.996017.

    """

    def __init__(self, space_fn, objectives, recombination=None, mutate_probability=0.7, population_size=30,
                 space_sample_validation_fn=None, random_state=None):
        survival = _RankAndCrowdSortSurvival(directions=[o.direction for o in objectives],
                                             population_size=population_size,
                                             random_state=random_state)

        super(NSGAIISearcher, self).__init__(space_fn=space_fn, objectives=objectives, survival=survival,
                                             recombination=recombination, mutate_probability=mutate_probability,
                                             space_sample_validation_fn=space_sample_validation_fn,
                                             random_state=random_state)


class _RDominanceSurvival(_RankAndCrowdSortSurvival):

    def __init__(self, directions, population_size, random_state, ref_point, weights, threshold):
        super(_RDominanceSurvival, self).__init__(directions, population_size=population_size, random_state=random_state)
        self.ref_point = ref_point
        self.weights = weights
        # enables the DM to control the selection pressure of the r-dominance relation.
        self.threshold = threshold

    def dominate(self, ind1: _NSGAIndividual, ind2: _NSGAIndividual, pop: List[_NSGAIndividual], directions=None):

        # check pareto dominate
        if pareto_dominate(ind1.scores, ind2.scores, directions=directions):
            return True

        if pareto_dominate(ind2.scores, ind1.scores, directions=directions):
            return False

        # in case of pareto-equivalent, compare distance
        scores = np.array([_.scores for _ in pop])
        scores_extend = np.max(scores, axis=0) - np.min(scores, axis=0)
        distances = []
        for indi in pop:
            # Calculate weighted Euclidean distance of two solution.
            # Note: if ref_point is infeasible value, distance maybe larger than 1
            indi.distance = np.sqrt(np.sum(np.square((np.asarray(indi.scores) - self.ref_point) / scores_extend) * self.weights))
            distances.append(indi.distance)

        dist_extent = np.max(distances) - np.min(distances)

        return (ind1.distance - ind2.distance) / dist_extent < -self.threshold

    def sort_font(self, front: List[_NSGAIndividual]):
        return sorted(front, key=lambda v: v.distance, reverse=False)

    def sort_population(self, population: List[_NSGAIndividual]):
        return sorted(population, key=cmp_to_key(self.cmp_operator), reverse=True)

    @staticmethod
    def cmp_operator(s1: _NSGAIndividual, s2: _NSGAIndividual):
        if s1.rank < s2.rank:
            return 1
        elif s1.rank == s2.rank:
            if s1.distance < s2.distance:  # the smaller the distance the better
                return 1
            elif s1.distance == s2.distance:
                return 0
            else:
                return -1
        else:
            return -1

    def __repr__(self):
        return f"{self.__class__.__name__}(ref_point={self.ref_point}, weights={self.weights}, " \
               f"threshold={self.threshold}, random_state={self.random_state})"


class RNSGAIISearcher(_NSGAIIBasedSearcher):
    """An implementation of R-NSGA-II which is a variant of NSGA-II algorithm.

    Parameters
    ----------
    space_fn: callable, required
        A search space function which when called returns a `HyperSpace` instance

    objectives: List[Objective], optional, (default to NumOfFeatures instance)
        The optimization objectives.

    ref_point: Tuple[float], required
        user-specified reference point, used to guide the search toward the desired region.

    weights:  Tuple[float], optional, default to uniform
        weights vector, provides more detailed information about what Pareto optimal to converge to.

    dominance_threshold: float, optional, default to 0.3
        distance threshold, in case of pareto-equivalent, compare distance between two solutions.

    recombination: Recombination, required
        the strategy to recombine DNA of parents to generate offspring. Builtin strategies:
        - ShuffleCrossOver
        - UniformCrossover
        - SinglePointCrossOver

    mutate_probability: float, optional, default to 0.7
        the probability of genetic variation for offspring, when the parents can not recombine,
        it will definitely mutate a gene for the generated offspring.

    population_size: int, default to 30
        size of population

    space_sample_validation_fn: callable or None, (default=None)
        used to verify the validity of samples from the search space, and can be used to add specific constraint
        rules to the search space to reduce the size of the space.

    random_state: np.RandomState, optional
        used to reproduce the search process

    References
    ----------
    [1] L. Ben Said, S. Bechikh and K. Ghedira, "The r-Dominance: A New Dominance Relation for Interactive Evolutionary Multicriteria Decision Making," in IEEE Transactions on Evolutionary Computation, vol. 14, no. 5, pp. 801-818, Oct. 2010, doi: 10.1109/TEVC.2010.2041060.
    """
    def __init__(self, space_fn, objectives, ref_point=None, weights=None, dominance_threshold=0.3,
                 recombination=None, mutate_probability=0.7, population_size=30,
                 space_sample_validation_fn=None, random_state=None):
        """
        """

        n_objectives = len(objectives)

        ref_point = ref_point if ref_point is not None else [0.0] * n_objectives
        weights = weights if weights is not None else [1 / n_objectives] * n_objectives
        directions = [o.direction for o in objectives]

        random_state if random_state is not None else get_random_state()

        survival = _RDominanceSurvival(random_state=random_state, population_size=population_size,
                                       ref_point=ref_point, weights=weights, threshold=dominance_threshold,
                                       directions=directions)

        super(RNSGAIISearcher, self).__init__(space_fn=space_fn, objectives=objectives, recombination=recombination,
                                              survival=survival, mutate_probability=mutate_probability,
                                              space_sample_validation_fn=space_sample_validation_fn,
                                              random_state=random_state)

    def _plot_population(self, figsize=(6, 6), show_ref_point=True, show_weights=False, **kwargs):
        from matplotlib import pyplot as plt

        def attach(ax):
            if show_ref_point:
                ref_point = self.survival.ref_point
                ax.scatter([ref_point[0]], [ref_point[1]], c='green', marker="*", label='ref point')
            if show_weights:
                weights = self.survival.weights
                # plot a vector
                ax.quiver(0, 0, weights[0], weights[1], angles='xy', scale_units='xy', label='weights')

        figs, axes = plt.subplots(2, 2, figsize=(figsize[0] * 2, figsize[0] * 2))
        historical_individuals = self.get_historical_population()

        # 1. ranking plot
        ax1 = axes[0][0]
        self._sub_plot_ranking(ax1, historical_individuals)
        attach(ax1)

        # 2. population plot
        ax2 = axes[0][1]
        self._sub_plot_pop(ax2, historical_individuals)
        attach(ax2)

        # 3. r-dominated plot
        ax3 = axes[1][0]
        n_set = self.get_nondominated_set()
        d_set: List[Individual] = list(filter(lambda v: v not in n_set, historical_individuals))
        self._do_plot(n_set, color='red', label='non-dominated', ax=ax3, marker="o")  # , marker="o"
        self._do_plot(d_set, color='blue', label='dominated', ax=ax3, marker="o")
        ax3.set_title(f"non-dominated solution (total={len(historical_individuals)}) in R-dominance scene")
        objective_names = [_.name for _ in self.objectives]
        ax3.set_xlabel(objective_names[0])
        ax3.set_ylabel(objective_names[1])
        ax3.legend()
        attach(ax3)

        # 4. pareto dominated plot
        ax4 = axes[1][1]
        self._plot_pareto(ax4, historical_individuals)
        attach(ax4)

        return figs, axes
