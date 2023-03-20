# -*- coding:utf-8 -*-
from typing import List

import numpy as np
from hypernets.core import HyperSpace, get_random_state
from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection, Searcher

from .genetic import Individual, ShuffleCrossOver, SinglePointCrossOver, UniformCrossover, SinglePointMutation
from .moo import pareto_dominate
from .moo import MOOSearcher


class Direction:
    def __init__(self, weight_vector: np.ndarray, random_state):

        self.weight_vector = weight_vector
        self.random_state= random_state
        self.neighbors = None
        self.individual = None
        self.candidate_dna = None

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def update_individual(self, individual: Individual):
        self.individual = individual

    def update_candidate_dna(self, candidate_dna: HyperSpace):
        self.candidate_dna = candidate_dna

    def random_select_neighbors(self, n):
        neighbor_len = len(self.neighbors)
        if n > neighbor_len:
            raise RuntimeError(f"required neighbors = {n} bigger that all neighbors = {neighbor_len} .")

        if n == 1:
            return [self.neighbors[self.random_state.randint(0, neighbor_len, size=n)[0]]]

        # group by params
        params_list = []
        for neighbor in self.neighbors:
            assert neighbor.dna.all_assigned
            params_list.append((frozenset([p.alias for p in neighbor.dna.get_assigned_params()]), neighbor))

        params_dict = {}
        for param in params_list:
            if not param[0] in params_dict:
                params_dict[param[0]] = [param[1]]
            else:
                params_dict[param[0]].append(param[1])

        for k, v in params_dict.items():
            if len(v) >= n:
                idx = self.random_state.randint(0, neighbor_len, size=n)
                return [self.neighbors[i].individual for i in idx]

        raise RuntimeError(f"required neighbors = {n} bigger that all neighbors = {neighbor_len} .")


class Decomposition:
    def __init__(self, **kwargs):
        pass

    def adaptive_normalization(self, F, ideal, nadir):
        """For objectives space normalization, the formula:
            f_{i}' = \frac{f_i - z_i^*}{z_i^{nad} - z^* + \epsilon }
        """
        eps = 1e-6
        return (F - ideal) / (nadir - ideal + eps)

    def do(self, scores: np.ndarray, weight_vector: np.ndarray, Z: np.ndarray, ideal: np.ndarray,
           nadir: np.ndarray, **kwargs):
        raise NotImplementedError

    def __call__(self, F: np.ndarray, weight_vector: np.ndarray, Z: np.ndarray, ideal: np.ndarray, nadir: np.ndarray):
        N_F = self.adaptive_normalization(F, ideal, nadir)
        return self.do(N_F, weight_vector, Z, ideal, nadir)


class WeightedSumDecomposition(Decomposition):

    def do(self, scores: np.ndarray, weight_vector, Z: np.ndarray, ideal: np.ndarray, nadir: np.ndarray, **kwargs):
        return (scores * weight_vector).sum()


class TchebicheffDecomposition(Decomposition):
    def do(self, scores: np.ndarray, weight_vector, Z, ideal:np.ndarray, nadir: np.ndarray, **kwargs):
        F = scores
        return np.max(np.abs(F - Z) * weight_vector)


class PBIDecomposition(Decomposition):

    def __init__(self, penalty=0.5):
        super().__init__()
        self.penalty = penalty

    def do(self, scores: np.ndarray, weight_vector: np.ndarray, Z: np.ndarray, ideal: np.ndarray, nadir: np.ndarray,
           **kwargs):
        """An implementation of "Boundary Intersection Approach base on penalty"
        :param scores
        :param weight_vector
        :param theta: Penalty F deviates from the weight vector.
        """
        F = scores
        d1 = ((F - Z) * weight_vector).sum() / np.linalg.norm(weight_vector)
        d2 = np.linalg.norm((Z + weight_vector * d1) - F)

        return d1 + d2 * self.penalty


class MOEADSearcher(MOOSearcher):
    """
    References
    ----------
        [1]. Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, Dec. 2007, doi: 10.1109/TEVC.2007.892759.
    """

    def __init__(self, space_fn, objectives, n_sampling=5, n_neighbors=2,
                 recombination=None, mutate_probability=0.3,
                 decomposition=None, decomposition_options=None,
                 optimize_direction=OptimizeDirection.Minimize, use_meta_learner=False,
                 space_sample_validation_fn=None, random_state=None):
        """
        :param space_fn:
        :param n_sampling: the number of samples in each objective
        :param n_objectives: number of objectives
        :param n_neighbors: number of neighbors to mating
        :param mutate_probability:
        :param decomposition: decomposition approach, default is None one of tchebicheff,weighted_sum, pbi
        :param optimize_direction:
        :param use_meta_learner:
        :param space_sample_validation_fn:
        :param random_state:
        """
        super(MOEADSearcher, self).__init__(space_fn=space_fn, objectives=objectives,
                                            optimize_direction=optimize_direction, use_meta_learner=use_meta_learner,
                                            space_sample_validation_fn=space_sample_validation_fn)

        if optimize_direction != OptimizeDirection.Minimize:
            raise ValueError("optimization towards maximization is not supported.")

        weight_vectors = self.init_mean_vector_by_NBI(n_sampling, self.n_objectives)  # uniform weighted vectors

        if random_state is None:
            self.random_state = get_random_state()
        else:
            self.random_state = np.random.RandomState(seed=random_state)

        self.mutation = SinglePointMutation(random_state=self.random_state, proba=mutate_probability)

        n_vectors = weight_vectors.shape[0]
        if n_neighbors > n_vectors:
            raise RuntimeError(f"n_neighbors should less that {n_vectors - 1}")

        if recombination is None:
            self.recombination = ShuffleCrossOver
        else:
            recombination_mapping = {
                'shuffle': ShuffleCrossOver,
                'uniform': UniformCrossover,
                'single_point': SinglePointCrossOver,
            }
            if recombination in recombination_mapping:
                self.recombination = recombination_mapping[recombination](random_state=self.random_state)
            else:
                raise RuntimeError(f'unseen recombination approach {decomposition}.')

        if decomposition_options is None:
            decomposition_options = {}

        if decomposition is None:
            decomposition_cls = TchebicheffDecomposition
        else:
            decomposition_mapping = {
                'tchebicheff': TchebicheffDecomposition,
                'weighted_sum': WeightedSumDecomposition,
                'pbi': PBIDecomposition
            }
            decomposition_cls = decomposition_mapping.get(decomposition)

            if decomposition_cls is None:
                raise RuntimeError(f'unseen decomposition approach {decomposition}.')

        self.decomposition = decomposition_cls(**decomposition_options)

        self.n_neighbors = n_neighbors
        self.directions = self.init_population(weight_vectors)
        logger.info(f"population size is {len(self.directions)}")

        self._pop_history = []  # to store all existed individuals

    @property
    def population_size(self):
        return len(self.directions)

    @property
    def n_objectives(self):
        return len(self.objectives)

    def distribution_number(self, n_samples, n_objectives):
        """Uniform weighted vectors, an implementation of Normal-boundary intersection.
            N = C_{n_samples+n_objectives-1}^{n_objectives-1}
            N  is the total num of generated vectors.
        :param n_samples: the number of samples in each objective
        :param n_objectives:
        :return:
        """
        if n_objectives == 1:
            return [[n_samples]]
        vectors = []
        for i in range(1, n_samples - (n_objectives - 1) + 1):
            right_vec = self.distribution_number(n_samples - i, n_objectives - 1)
            a = [i]
            for item in right_vec:
                vectors.append(a + item)
        return vectors

    def init_mean_vector_by_NBI(self, n_samples, n_objectives):
        # Das I, Dennis J E. "Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems[J]." SIAM Journal on Optimization, 1998, 8(3): 631-657.
        vectors = self.distribution_number(n_samples + n_objectives, n_objectives)
        vectors = (np.array(vectors) - 1) / n_samples
        return vectors

    def calc_euler_distance(self, vectors):
        v_len = len(vectors)
        Euler_distance = np.zeros((v_len, v_len))
        for i in range(v_len):
            for j in range(v_len):
                distance = ((vectors[i] - vectors[j]) ** 2).sum()
                Euler_distance[i][j] = distance
        return Euler_distance

    def init_population(self, weight_vectors):
        pop_size = len(weight_vectors)
        directions = []
        for i in range(pop_size):
            weight_vector = weight_vectors[i]
            directions.append(Direction(weight_vector=weight_vector, random_state=self.random_state))
            # space_sample = self._sample_and_check(self._random_sample)
            # pop.append(MOEADIndividual(dna=space_sample, weight_vector=weight_vector, random_state=self.random_state))

        # euler distances matrix
        euler_distances = self.calc_euler_distance(weight_vectors)

        # calc neighbors of each vector
        for i in range(pop_size):
            sorted_distances = np.argsort(euler_distances[i])
            neighbors = [directions[sorted_distances[i]] for i in range(self.n_neighbors)]
            directions[i].set_neighbors(neighbors)

        return directions

    def get_nondominated_set(self):
        population = self.get_historical_population()

        def find_non_dominated_solu(indi):
            if (np.array(indi.scores) == None).any():  # illegal individual for the None scores
                return False
            for indi_ in population:
                if indi_ == indi:
                    continue
                return pareto_dominate(x1=indi_.scores, x2=indi.scores, directions=self.directions)
            return True  # this is a pareto optimal

        # find non-dominated solution for every solution
        ns = list(filter(lambda s: find_non_dominated_solu(s), population))
        return ns

    def sample(self):
        # random sample
        for direction in self.directions:
            if direction.individual is None:
                sample = self._sample_and_check(self._random_sample)
                return sample

        # select the sub-objective
        direction_inx = len(self.directions) % len(self.directions)
        direction = self.directions[direction_inx]

        # select neighbors to  crossover and mutate
        try:
            offspring = self.recombination(*direction.random_select_neighbors(2), self.space_fn())
            # sine the length between sample space does not match usually cause no enough neighbors
            logger.info("do recombination.")
            MP = self.mutation.proba
        except Exception as e:
            offspring = direction.random_select_neighbors(1)[0].individual.dna
            # Must mutate because of failing crossover
            MP = 1
        final_offspring = self.mutation.do(offspring, self.space_fn(), proba=MP)

        direction.offspring = final_offspring

        if final_offspring is not None:
            return self._sample_and_check(lambda: final_offspring)
        else:
            space_sample = self._sample_and_check(self._random_sample)
            return space_sample

    def get_best(self):
        return list(map(lambda s: s[0], self.get_nondominated_set()))

    def get_reference_point(self):
        """calculate Z in tchebicheff decomposition
        """
        scores_list = []
        for solu in self._pop_history:
            scores_list.append(solu.scores)
        return np.min(np.array(scores_list), axis=0)

    def get_ideal_point(self):
        non_dominated = self.get_nondominated_set()
        return np.min(np.array(list(map(lambda _: _.scores, non_dominated))), axis=0)

    def get_nadir_point(self):
        non_dominated = self.get_nondominated_set()
        return np.max(np.array(list(map(lambda _: _.scores, non_dominated))), axis=0)

    def _update_neighbors(self, direction: Direction, candidate: Individual):
        for neighbor in direction.neighbors:
            neighbor: Direction = neighbor
            wv = neighbor.weight_vector
            Z = self.get_reference_point()
            nadir = self.get_nadir_point()
            ideal = self.get_ideal_point()

            if (self.decomposition(neighbor.individual.scores, wv, Z, ideal, nadir) - self.decomposition(candidate.scores, wv, Z, ideal, nadir)) > 0:
                neighbor.update_individual(candidate)

    def update_result(self, space, result):
        assert space
        individual = Individual(dna=space, scores=np.array(result), random_state=self.random_state)
        self._pop_history.append(individual)

        if len(self._pop_history) == self.population_size:
            # init population
            for direction, indi in zip(self.directions, self._pop_history):
                direction.update_individual(indi)
        elif len(self._pop_history) > self.population_size:
            for direction in self.directions:
                if direction.candidate_dna == space:
                    self._update_neighbors(direction, individual)
        else:
            pass  # init population needs enough individual

    def get_historical_population(self) -> List[Individual]:
        return self._pop_history

    def get_population(self) -> List[Individual]:
        raise NotImplementedError

    @property
    def parallelizable(self):
        return False
