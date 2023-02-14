from typing import List
from functools import cmp_to_key

import numpy as np

from .mo import dominate, calc_nondominated_set
from ..core import HyperSpace, Searcher, OptimizeDirection, get_random_state
from .genetic import Recombination, Individual, SinglePointMutation


class NSGAIndividual(Individual):
    def __init__(self, dna: HyperSpace, scores: np.ndarray, random_state):

        super().__init__(dna, scores, random_state)

        self.dna = dna
        self.scores = scores

        self.rank: int = -1  # rank starts from 1

        self.S: List[NSGAIndividual] = []
        self.n: int = -1

        self.distance = 0  # crowding-distance


class NSGAIISearcher(Searcher):
    """
    References:
        [1]. K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, April 2002, doi: 10.1109/4235.996017.
    """

    def __init__(self, space_fn, recombination=None, mutate_probability=0.7, population_size=30,
                 optimize_direction=OptimizeDirection.Minimize, use_meta_learner=False,
                 space_sample_validation_fn=None, random_state=None):
        """
        :param space_fn:
        :param mutate_probability:
        :param optimize_direction:
        :param use_meta_learner:
        :param space_sample_validation_fn:
        :param random_state:
        """
        Searcher.__init__(self, space_fn=space_fn, optimize_direction=optimize_direction,
                          use_meta_learner=use_meta_learner, space_sample_validation_fn=space_sample_validation_fn)

        self.population: List[NSGAIndividual] = []
        self.random_state = random_state if random_state is not None else get_random_state()
        self.recombination: Recombination = recombination

        self.mutation = SinglePointMutation(self.random_state, mutate_probability)

        self.population_size = population_size

    @staticmethod
    def fast_non_dominated_sort(P: List[NSGAIndividual]):

        F_1 = []
        F = [F_1]  # to store pareto front of levels respectively

        for p in P:
            S_p = []
            n_p = 0
            for q in P:
                if p == q:
                    continue
                if dominate(p.scores, q.scores):
                    S_p.append(q)
                if dominate(q.scores, p.scores):
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

    @staticmethod
    def crowding_distance_assignment(I: List[NSGAIndividual]):
        scores_array = np.array([indi.scores for indi in I])

        maximum_array = np.max(scores_array, axis=0)
        minimum_array = np.min(scores_array, axis=0)

        for m in range(len(I[0].scores)):
            sorted_I = list(sorted(I, key=lambda v: v.scores[m], reverse=False))
            sorted_I[0].distance = float("inf")  # so that boundary points always selected, because they are not crowd
            sorted_I[len(I)-1].distance = float("inf")
            # only assign distances for non-boundary points
            for i in range(1, (len(I) - 1)):
                sorted_I[i].distance = sorted_I[i].distance\
                                + (sorted_I[i+1].scores[m] - sorted_I[i - 1].scores[m]) \
                                / (maximum_array[m] - minimum_array[m])
        return I

    def binary_tournament_select(self, population):
        indi_inx = self.random_state.randint(low=0, high=len(population) - 1, size=2)

        p1 = population[indi_inx[0]]
        p2 = population[indi_inx[1]]

        # select the first parent
        if self.compare_solution(p1, p2) >= 0:
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

        if self.compare_solution(p1, p2) >= 0:
            second_inx = indi_inx[0]
        else:
            second_inx = indi_inx[1]

        return population[first_inx], population[second_inx]

    def compare_solution(self, s1: NSGAIndividual, s2: NSGAIndividual):
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

    def sample(self):
        if len(self.population) < self.population_size:
            return self._sample_and_check(self._random_sample)

        P_sorted = self.fast_non_dominated_sort(self.population)
        P_selected:List[NSGAIndividual] = []

        rank = 0
        while len(P_selected) + len(P_sorted[rank]) <= self.population_size:
            individuals = self.crowding_distance_assignment(P_sorted[rank])
            P_selected.extend(individuals)
            rank = rank+1
            if rank >= len(P_sorted):  # no enough elements
                break

        # ensure population size
        P_final = list(sorted(P_selected, key=cmp_to_key(self.compare_solution)))[:self.population_size]

        # binary tournament selection operation
        p1, p2 = self.binary_tournament_select(P_final)

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
        self.population.append(NSGAIndividual(space, np.array(result), self.random_state))

    def reset(self):
        pass

    def export(self):
        pass
