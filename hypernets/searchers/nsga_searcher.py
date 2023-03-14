from typing import List
from functools import cmp_to_key

import numpy as np

from .moo import MOOSearcher, pareto_dominate, calc_nondominated_set
from ..core import HyperSpace, Searcher, OptimizeDirection, get_random_state
from .genetic import Recombination, Individual, SinglePointMutation, Survival


class NSGAIndividual(Individual):
    def __init__(self, dna: HyperSpace, scores: np.ndarray, random_state):

        super().__init__(dna, scores, random_state)

        self.dna = dna
        self.scores = scores

        self.rank: int = -1  # rank starts from 1

        self.S: List[NSGAIndividual] = []
        self.n: int = -1

        self.distance = 0  # crowding-distance


def cmp_operator(s1: NSGAIndividual, s2: NSGAIndividual):
    if s1.rank < s2.rank:
        return 1
    elif s1.rank == s2.rank:
        if s1.distance > s2.distance:
            return 1
        elif s1.distance == s2.distance:
            return 0
        else:
            return -1
    else:
        return -1


class RankAndCrowdSortSurvival(Survival):

    def __init__(self, random_state):
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

    def fast_non_dominated_sort(self, P: List[NSGAIndividual], directions):

        F_1 = []
        F = [F_1]  # to store pareto front of levels respectively

        for p in P:
            S_p = []
            n_p = 0
            for q in P:
                if p == q:
                    continue
                if self.dominate(p.scores, q.scores, pop=P, directions=directions):
                    S_p.append(q)
                if self.dominate(q.scores, p.scores, pop=P,  directions=directions):
                    n_p = n_p + 1

            p.S = S_p
            p.n = n_p

            if n_p == 0:
                p.rank = 1
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

    def update(self, pop: List[NSGAIndividual], pop_size: int, challengers: List[Individual], directions: List[str]):
        temp_pop = []
        temp_pop.extend(pop)
        temp_pop.extend(challengers)

        P_sorted = self.fast_non_dominated_sort(temp_pop, directions)
        P_selected: List[NSGAIndividual] = []

        rank = 0
        while len(P_selected) + len(P_sorted[rank]) <= pop_size:
            individuals = self.crowding_distance_assignment(P_sorted[rank])
            P_selected.extend(individuals)
            rank = rank + 1
            if rank >= len(P_sorted):  # no enough elements
                break

        # ensure population size
        P_final = list(sorted(P_selected, key=cmp_to_key(cmp_operator)))[:pop_size]
        return P_final

    def dominate(self, x1: np.ndarray, x2: np.ndarray, pop, directions=None):
        return pareto_dominate(x1=x1, x2=x2, directions=directions)


class NSGAIISearcher(MOOSearcher):
    """
    References:
        [1]. K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, April 2002, doi: 10.1109/4235.996017.
    """

    def __init__(self, space_fn, objectives, recombination=None, mutate_probability=0.7,
                 population_size=30, use_meta_learner=False, space_sample_validation_fn=None, random_state=None):
        """
        :param space_fn:
        :param mutate_probability:
        :param optimize_direction:
        :param use_meta_learner:
        :param space_sample_validation_fn:
        :param random_state:
        """
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
        return RankAndCrowdSortSurvival(self.random_state)

    def binary_tournament_select(self, population):
        indi_inx = self.random_state.randint(low=0, high=len(population) - 1, size=2)  # fixme: maybe duplicated inx

        p1 = population[indi_inx[0]]
        p2 = population[indi_inx[1]]

        # select the first parent
        if cmp_operator(p1, p2) >= 0:
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

        if cmp_operator(p1, p2) >= 0:
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

    def get_nondominated_set(self):
        return calc_nondominated_set(self.population)

    def update_result(self, space, result):
        indi = NSGAIndividual(space, np.array(result), self.random_state)
        self._historical_individuals.append(indi)  # add to history
        p = self.survival.update(pop=self.population, pop_size=self.population_size,
                                 challengers=[indi], directions=self.directions)
        self.population = p

    def get_historical_population(self):
        return self._historical_individuals

    def get_population(self) -> List[Individual]:
        return self.population

    def reset(self):
        pass

    def export(self):
        pass


class RDominanceSurvival(RankAndCrowdSortSurvival):

    def __init__(self, random_state, ref_point, weights, dominance_threshold):
        super(RDominanceSurvival, self).__init__(random_state)
        self.ref_point = ref_point
        self.weights = weights
        self.dominance_threshold = dominance_threshold

    @staticmethod
    def r_dominance(x, y, ref_point, weights, directions, pop: np.ndarray, threshold):
        def distance(x: np.ndarray, g: np.ndarray, w: np.ndarray, scores_diff: np.ndarray):
            """Calculate weighted Euclidean distance of two solution.
                :param x: considered solution
                :param g: user-specified reference point
                :param w: weight vector
                :param scores_diff: max(obj_i) - min(obj_i), for normalization
            """
            return np.sum(np.sqrt(np.square((x - g) / scores_diff) * w))

        assert pop.ndim == 2
        if pareto_dominate(x, y, directions):
            return True

        scores_diff = np.max(pop, axis=0) - np.min(pop, axis=0)

        distances = [distance(_, ref_point, weights, scores_diff) for _ in pop]
        d = (distance(x, ref_point, weights, scores_diff) - distance(y, ref_point, weights, scores_diff)) / (np.max(distances) - np.min(distances))
        return d < -threshold

    def dominate(self, x1: np.ndarray, x2: np.ndarray, pop, directions=None):
        pop_scores = np.array([_.scores for _ in pop])
        return self.r_dominance(x=x1, y=x2, ref_point=self.ref_point, weights=self.weights, directions=directions,
                                pop=pop_scores, threshold=self.dominance_threshold)


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
        self.epsilon = epsilon

        super().__init__(space_fn=space_fn, objectives=objectives,recombination=recombination,
                         mutate_probability=mutate_probability, population_size=population_size,
                         use_meta_learner=use_meta_learner,
                         space_sample_validation_fn=space_sample_validation_fn, random_state=random_state)

    def create_survival(self):
        return RDominanceSurvival(random_state=self.random_state, ref_point=self.ref_point,
                                  weights=self.weights, dominance_threshold=self.dominance_threshold)
