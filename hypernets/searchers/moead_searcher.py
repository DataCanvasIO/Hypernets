# -*- coding:utf-8 -*-
from typing import List

import numpy as np
from hypernets.core import HyperSpace, get_random_state
from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection
from hypernets.core import pareto

from .genetic import Individual, ShuffleCrossOver, SinglePointCrossOver, UniformCrossover, SinglePointMutation, \
    create_recombination
from .moo import MOOSearcher
from ..utils import const


class _Direction:
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
        return [self.neighbors[i] for i in self.random_state.randint(0, neighbor_len, size=n)]

        # group by params
        # params_list = []
        # for neighbor in self.neighbors:
        #     assert neighbor.individual.dna.all_assigned
        #     params_list.append((frozenset([p.alias for p in neighbor.individual.dna.get_assigned_params()]), neighbor))
        #
        # params_dict = {}
        # for param in params_list:
        #     if not param[0] in params_dict:
        #         params_dict[param[0]] = [param[1]]
        #     else:
        #         params_dict[param[0]].append(param[1])
        #
        # for k, v in params_dict.items():
        #     if len(v) >= n:
        #         idx = self.random_state.randint(0, neighbor_len, size=n)
        #         return [self.neighbors[i].individual for i in idx]


class Decomposition:
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def adaptive_normalization(F, ideal, nadir):
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
    """An implementation of "Boundary Intersection Approach base on penalty"

    Parameters
    ----------
    penalty: float, optional, default to 0.5
        Penalty the solution(F) deviates from the weight vector, the larger the value,
        the faster the convergence.
    """

    def __init__(self, penalty=0.5):
        super().__init__()
        self.penalty = penalty

    def do(self, scores: np.ndarray, weight_vector: np.ndarray, Z: np.ndarray, ideal: np.ndarray, nadir: np.ndarray,
           **kwargs):
        F = scores
        d1 = ((F - Z) * weight_vector).sum() / np.linalg.norm(weight_vector)
        d2 = np.linalg.norm((Z + weight_vector * d1) - F)

        return d1 + d2 * self.penalty

    def __repr__(self):
        return f"{self.__class__.__name__}(penalty={self.penalty})"


class MOEADSearcher(MOOSearcher):
    """An implementation of "MOEA/D".

    Parameters
    ----------
    space_fn: callable, required
        A search space function which when called returns a `HyperSpace` instance

    objectives: List[Objective], optional, (default to NumOfFeatures instance)
        The optimization objectives.

    n_sampling: int, optional, default to 5.
        The number of samples in each objective, it affects the number of optimization objectives after decomposition:

        :math:`N = C_{samples + objectives - 1}^{ objectives - 1 }`

    n_neighbors: int, optional, default to 3.
        Number of neighbors to crossover.

    recombination: Recombination, optional, default to instance of SinglePointCrossOver
        the strategy to recombine DNA of parents to generate offspring. Builtin strategies:

        - ShuffleCrossOver
        - UniformCrossover
        - SinglePointCrossOver

    decomposition: Decomposition, optional, default to instance of TchebicheffDecomposition

        The strategy to decompose multi-objectives optimization problem and calculate scores for the sub problem, now supported:

        - TchebicheffDecomposition
        - PBIDecomposition
        - WeightedSumDecomposition

        Due to the possible differences in dimension of objectives, normalization will be performed on the scores, the  formula:

        :math:`f_{i}' = \\frac{ f_i - z_i^* } { z_i ^ {nad} - z^* + \\epsilon }`


    mutate_probability: float, optional, default to 0.7
        the probability of genetic variation for offspring, when the parents can not recombine,
        it will definitely mutate a gene for the generated offspring.

    space_sample_validation_fn: callable or None, (default=None)
        used to verify the validity of samples from the search space, and can be used to add specific constraint
        rules to the search space to reduce the size of the space.

    random_state: np.RandomState, optional
        used to reproduce the search process


    References
    ----------
    [1] Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, Dec. 2007, doi: 10.1109/TEVC.2007.892759.

    [2] Das I, Dennis J E. "Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems[J]." SIAM Journal on Optimization, 1998, 8(3): 631-657.
    """

    def __init__(self, space_fn, objectives, n_sampling=5, n_neighbors=2, recombination=None, mutate_probability=0.7,
                 decomposition=None, space_sample_validation_fn=None, random_state=None):

        super(MOEADSearcher, self).__init__(space_fn=space_fn, objectives=objectives,
                                            optimize_direction=objectives[0].direction, use_meta_learner=False,
                                            space_sample_validation_fn=space_sample_validation_fn)
        for o in objectives:
            if o.direction != OptimizeDirection.Minimize.value:
                raise ValueError(f"optimization towards maximization is not supported, objective is {o}")

        weight_vectors = self.init_mean_vector_by_NBI(n_sampling, self.n_objectives)  # uniform weighted vectors

        if random_state is None:
            self.random_state = get_random_state()
        else:
            self.random_state = random_state

        self.mutation = SinglePointMutation(random_state=self.random_state, proba=mutate_probability)

        n_vectors = weight_vectors.shape[0]
        if n_neighbors > n_vectors:
            raise RuntimeError(f"n_neighbors should less that {n_vectors - 1}")

        if recombination is None:
            self.recombination = create_recombination(const.COMBINATION_SINGLE_POINT, random_state=self.random_state)
        else:
            self.recombination = recombination

        if decomposition is None:
            self.decomposition = create_decomposition(const.DECOMPOSITION_TCHE)
        else:
            self.decomposition = decomposition

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
        """
        if n_objectives == 1:
            return [[n_samples]]
        vectors = []
        for i in range(n_samples - (n_objectives - 1)):
            right_vec = self.distribution_number(n_samples - (i + 1), n_objectives - 1)
            a = [i+1]
            for item in right_vec:
                vectors.append(a + item)
        return vectors

    def init_mean_vector_by_NBI(self, n_samples, n_objectives):
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
            directions.append(_Direction(weight_vector=weight_vector, random_state=self.random_state))
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

    def sample(self, space_options=None):
        for direction in self.directions:
            if direction.individual is None:
                sample = self._sample_and_check(self._random_sample)
                return sample

        # select the sub-objective
        direction_inx = len(self.directions) % len(self.directions)
        direction = self.directions[direction_inx]

        # select neighbors to  crossover and mutate
        direction1, direction2 = direction.random_select_neighbors(2)
        ind1 = direction1.individual
        ind2 = direction2.individual
        if self.recombination.check_parents(ind1, ind2):
            offspring = self.recombination(ind1, ind2, self.space_fn())
            # sine the length between sample space does not match usually cause no enough neighbors
            # logger.info("do recombination.")
            MP = self.mutation.proba
        else:
            offspring = direction.random_select_neighbors(1)[0].individual.dna
            MP = 1  # Must mutate because of failing crossover

        final_offspring = self.mutation.do(offspring, self.space_fn(), proba=MP)

        direction.offspring = final_offspring

        if final_offspring is not None:
            return self._sample_and_check(lambda: final_offspring)
        else:
            space_sample = self._sample_and_check(self._random_sample)
            return space_sample

    def get_nondominated_set(self):
        population = self.get_historical_population()

        scores = np.array([_.scores for _ in population])
        obj_directions = [_.direction for _ in self.objectives]

        non_dominated_inx = pareto.calc_nondominated_set(scores, directions=obj_directions)

        return [population[i] for i in non_dominated_inx]

    def _plot_population(self, figsize=(6, 6), **kwargs):
        from matplotlib import pyplot as plt

        figs, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
        historical_individuals = self.get_historical_population()

        # 1. population plot
        self._sub_plot_pop(axes[0], historical_individuals)

        # 2. pareto dominated plot
        self._plot_pareto(axes[1], historical_individuals)

        return figs, axes

    def get_best(self):
        return list(map(lambda _: _.dna, self.get_nondominated_set()))

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

    def _update_neighbors(self, direction: _Direction, candidate: Individual):
        for neighbor in direction.neighbors:
            neighbor: _Direction = neighbor
            wv = neighbor.weight_vector
            Z = self.get_reference_point()
            nadir = self.get_nadir_point()
            ideal = self.get_ideal_point()

            if (self.decomposition(neighbor.individual.scores, wv, Z, ideal, nadir) - self.decomposition(candidate.scores, wv, Z, ideal, nadir)) > 0:
                neighbor.update_individual(candidate)

    def update_result(self, space, result):
        assert space
        individual = Individual(dna=space, scores=result, random_state=self.random_state)
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
        return list(map(lambda d: d.individual,
                        filter(lambda v: v.individual is not None, self.directions)))

    @property
    def parallelizable(self):
        return False

    def reset(self):
        pass

    def export(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(objectives={self.objectives}, n_neighbors={self.n_neighbors}," \
               f" recombination={self.recombination}, " \
               f"mutation={self.mutation}, population_size={self.population_size})"


def create_decomposition(name, **kwargs):
    if name == const.DECOMPOSITION_TCHE:
        return TchebicheffDecomposition()
    elif name == const.DECOMPOSITION_WS:
        return WeightedSumDecomposition()
    elif name == const.DECOMPOSITION_PBI:
        return PBIDecomposition(**kwargs)
    else:
        raise RuntimeError(f'unseen decomposition approach {name}.')
