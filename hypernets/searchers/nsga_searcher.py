from typing import List
from functools import cmp_to_key, partial

import numpy as np
from hypernets.utils import logging as hyn_logging

from .moo import MOOSearcher, pareto_dominate
from ..core import HyperSpace, Searcher, OptimizeDirection, get_random_state
from .genetic import Recombination, Individual, SinglePointMutation, Survival

logger = hyn_logging.get_logger(__name__)


class NSGAIndividual(Individual):
    def __init__(self, dna: HyperSpace, scores: np.ndarray, random_state):

        super().__init__(dna, scores, random_state)

        self.dna = dna
        self.scores = scores

        self.rank: int = -1  # rank starts from 1

        self.S: List[NSGAIndividual] = []
        self.n: int = -1

        self.T: List[NSGAIndividual] = []

        self.distance: float = -1.0  # crowding-distance

    def reset(self):
        self.rank = -1
        self.S = []
        self.n = 0
        self.T = []
        self.distance = -1.0

    def __repr__(self):
        return f"(scores={self.scores}, rank={self.rank}, n={self.n}, distance={self.distance})"


class RankAndCrowdSortSurvival(Survival):

    def __init__(self, directions, population_size, random_state):
        self.directions = directions
        self.population_size = population_size
        self.random_state = random_state

    @staticmethod
    def crowding_distance_assignment(I: List[NSGAIndividual]):
        scores_array = np.array([indi.scores for indi in I])

        maximum_array = np.max(scores_array, axis=0)
        minimum_array = np.min(scores_array, axis=0)

        for m in range(len(I[0].scores)):
            sorted_I = list(sorted(I, key=lambda v: v.scores[m], reverse=False))
            sorted_I[0].distance = float("inf")  # so that boundary points always selected, because they are not crowd
            sorted_I[len(I) - 1].distance = float("inf")
            # only assign distances for non-boundary points
            for i in range(1, (len(I) - 1)):
                sorted_I[i].distance = sorted_I[i].distance \
                                       + (sorted_I[i + 1].scores[m] - sorted_I[i - 1].scores[m]) \
                                       / (maximum_array[m] - minimum_array[m])
        return I

    def fast_non_dominated_sort(self, pop: List[NSGAIndividual]):
        for p in pop:
            p.reset()
        directions = self.directions
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

    def update(self, pop: List[NSGAIndividual], challengers: List[NSGAIndividual]):
        temp_pop = []
        temp_pop.extend(pop)
        temp_pop.extend(challengers)

        if len(pop) < self.population_size:
            return temp_pop

        p_sorted = self.fast_non_dominated_sort(temp_pop)
        if len(p_sorted) == 1 and len(p_sorted[0]) == 0:
            print(p_sorted)

        p_selected: List[NSGAIndividual] = []
        for rank, P_front in enumerate(p_sorted):
            if len(P_front) == 0:
                break

            individuals = self.crowding_distance_assignment(P_front)
            p_selected.extend(individuals)
            if len(p_selected) >= self.population_size:
                break

        # ensure population size
        p_cmp_sorted = list(sorted(p_selected, key=cmp_to_key(self.cmp_operator), reverse=True))
        p_final = p_cmp_sorted[:self.population_size]
        logger.debug(f"Individual have been removed from population: {p_cmp_sorted[self.population_size-1: ]}, sorted={p_cmp_sorted}")
        if challengers[0] in p_final:
            logger.debug(f"New individual{challengers[0]} goes into population, current pop {pop}")
        else:
            logger.debug(f"New individual{challengers[0]} does not go into population, current pop {pop}")

        return p_final

    def dominate(self, ind1: NSGAIndividual, ind2: NSGAIndividual, pop: List[NSGAIndividual]):
        return pareto_dominate(x1=ind1.scores, x2=ind2.scores, directions=self.directions)

    @staticmethod
    def cmp_operator(s1: NSGAIndividual, s2: NSGAIndividual):
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


class NSGAIISearcher(MOOSearcher):
    """
    References:
        [1]. K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, April 2002, doi: 10.1109/4235.996017.
    """

    def __init__(self, space_fn, objectives, recombination=None, mutate_probability=0.7,
                 population_size=30, use_meta_learner=False, space_sample_validation_fn=None, random_state=None):
        super().__init__(space_fn=space_fn, objectives=objectives, use_meta_learner=use_meta_learner,
                         space_sample_validation_fn=space_sample_validation_fn)

        self.population: List[NSGAIndividual] = []
        self.random_state = random_state if random_state is not None else get_random_state()
        self.recombination: Recombination = recombination

        self.mutation = SinglePointMutation(self.random_state, mutate_probability)

        self.population_size = population_size

        self.survival = self.create_survival()
        self._historical_individuals: List[NSGAIndividual] = []

    def create_survival(self):
        return RankAndCrowdSortSurvival(directions=self.directions,
                                        population_size=self.population_size,
                                        random_state=self.random_state)

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

    def sample(self):
        if len(self.population) < self.population_size:
            return self._sample_and_check(self._random_sample)

        # binary tournament selection operation
        p1, p2 = self.binary_tournament_select(self.population)

        try:
            offspring = self.recombination.do(p1, p2, self.space_fn())
            final_offspring = self.mutation.do(offspring.dna, self.space_fn())
        except Exception:
            final_offspring = self.mutation.do(p1.dna, self.space_fn(), proba=1)

        return final_offspring

    def get_best(self):
        return list(map(lambda v: v.dna, self.get_nondominated_set()))

    def update_result(self, space, result):
        indi = NSGAIndividual(space, np.array(result), self.random_state)
        self._historical_individuals.append(indi)  # add to history
        p = self.survival.update(pop=self.population,  challengers=[indi])
        self.population = p

    def plot_addition(self, ax, fig, **kwargs):
        p_sorted = self.survival.fast_non_dominated_sort(self.get_population())
        colors = ['c', 'm', 'y', 'r', 'g', 'b' ]
        for i, front in enumerate(p_sorted):
            scores = np.array([_.scores for _ in front])
            c_i = len(colors) - 1 if i > len(colors) - 1 else i
            ax.plot(scores[:, 0], scores[:, 1], color=colors[c_i], label=f"rank={i}")

        return ax, fig

    def get_nondominated_set(self):
        population = self.get_historical_population()

        def find_non_dominated_solu(indi):
            if (np.array(indi.scores) == None).any():  # illegal individual for the None scores
                return False
            for indi_ in population:
                if indi_ == indi:
                    continue
                if self.survival.dominate(ind1=indi_, ind2=indi, pop=population):
                    return False
            return True  # this is a pareto optimal

        # find non-dominated solution for every solution
        ns = list(filter(lambda s: find_non_dominated_solu(s), population))

        return ns

    def get_historical_population(self):
        return self._historical_individuals

    def get_population(self) -> List[Individual]:
        return self.population

    def reset(self):
        pass

    def export(self):
        pass


class RDominanceSurvival(RankAndCrowdSortSurvival):

    def __init__(self, directions, population_size, random_state, ref_point, weights, threshold):
        """ Calculate weighted Euclidean distance of two solution.

        Parameters
        ----------
        ref_point: user-specified reference point, note that since g is infeasible value, distance maybe larger than 1
        weights: weight vector
        threshold: distance threshold
        """

        super(RDominanceSurvival, self).__init__(directions, population_size=population_size, random_state=random_state)
        self.ref_point = ref_point
        self.weights = weights
        # enables the DM to control the selection pressure of the r-dominance relation.
        self.dominance_threshold = threshold

    def dominate(self, ind1: NSGAIndividual, ind2: NSGAIndividual, pop: List[NSGAIndividual], directions=None):

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
            indi.distance = np.sqrt(np.sum(np.square((indi.scores - self.ref_point) / scores_extend) * self.weights))
            distances.append(indi.distance)

        dist_extent = np.max(distances) - np.min(distances)

        if (ind1.distance - ind2.distance) / dist_extent < -self.dominance_threshold:
            return True
        else:
            return False

    @staticmethod
    def sort_within_font(front: List[NSGAIndividual]):
        return sorted(front, key=lambda v: v.distance, reverse=False)

    def update(self, pop: List[NSGAIndividual], challengers: List[Individual]):
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
        p_selected: List[NSGAIndividual] = []
        for rank, P_front in enumerate(p_sorted):
            if len(P_front) == 0:
                break
            individuals = self.sort_within_font(P_front)
            p_selected.extend(individuals)
            if len(p_selected) >= self.population_size:
                break

        # ensure population size
        p_final = list(sorted(p_selected, key=cmp_to_key(self.cmp_operator)))[:self.population_size]

        return p_final

    @staticmethod
    def cmp_operator(s1: NSGAIndividual, s2: NSGAIndividual):
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


class RNSGAIISearcher(NSGAIISearcher):
    """An implementation of R-NSGA-II which is a variant of NSGA-II algorithm.

        References:
            [1]. L. Ben Said, S. Bechikh and K. Ghedira, "The r-Dominance: A New Dominance Relation for Interactive Evolutionary Multicriteria Decision Making," in IEEE Transactions on Evolutionary Computation, vol. 14, no. 5, pp. 801-818, Oct. 2010, doi: 10.1109/TEVC.2010.2041060.
    """
    def __init__(self, space_fn, objectives, ref_point=None, weights=None, dominance_threshold=0.0001,
                 recombination=None, mutate_probability=0.7, epsilon=0.001, population_size=30, use_meta_learner=False,
                 space_sample_validation_fn=None, random_state=None):

        n_objectives = len(objectives)

        self.ref_point = ref_point if ref_point is not None else [0.0] * n_objectives
        # default is uniform weights
        self.weights = weights if weights is not None else [1 / n_objectives] * n_objectives
        self.dominance_threshold = dominance_threshold

        super(RNSGAIISearcher, self).__init__(space_fn=space_fn, objectives=objectives, recombination=recombination,
                                              mutate_probability=mutate_probability, population_size=population_size,
                                              use_meta_learner=use_meta_learner,
                                              space_sample_validation_fn=space_sample_validation_fn,
                                              random_state=random_state)

    def create_survival(self):
        return RDominanceSurvival(random_state=self.random_state,
                                  population_size=self.population_size,
                                  ref_point=self.ref_point,
                                  weights=self.weights, threshold=self.dominance_threshold,
                                  directions=self.directions)

    def plot_addition(self, ax, fig, show_ref_point=True, show_weights=False, **kwargs):
        if show_ref_point:
            ref_point = self.ref_point
            ax.scatter([ref_point[0]], [ref_point[1]], c='green', marker="*", label='ref point')

        if show_weights:
            weights = self.weights
            # plot a vector
            ax.quiver(0, 0, weights[0], weights[1], angles='xy', scale_units='xy', label='weights')

        p_sorted = self.survival.fast_non_dominated_sort(self.get_population())
        colors = ['c', 'm', 'y', 'r', 'g', 'b' ]

        for i, front in enumerate(p_sorted):
            scores = np.array([_.scores for _ in front])
            i_color = len(colors) - 1 if i > len(colors) - 1 else i
            ax.plot(scores[:, 0], scores[:, 1], color=colors[i_color], label=f"rank={i}")

        return ax, fig
