# -*- coding:utf-8 -*-
import numpy as np

from hypernets.core import HyperSpace, get_random_state
from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection, Searcher


class Individual:

    def __init__(self, dna, weight_vector, random_state):
        self.dna = dna
        self.weight_vector = weight_vector
        self.random_state = random_state

        self.neighbors = None
        self._offspring = None
        self._scores = None

    def get_offspring(self):
        return self._offspring

    def set_offspring(self, offspring):
        self._offspring = offspring

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def get_scores(self):
        return self._scores

    def set_scores(self, scores):
        self._scores = scores

    def update_dna(self, dna, scores):
        self.dna = dna
        self._scores = scores

    def random_neighbors(self, n):
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
                return [self.neighbors[i] for i in idx]

        raise RuntimeError(f"required neighbors = {n} bigger that all neighbors = {neighbor_len} .")


class Recombination:

    def __init__(self, random_state=None):
        if random_state is None:
            self.random_state = get_random_state()
        else:
            self.random_state = random_state

    def do(self, ind1: Individual, ind2: Individual, out_space: HyperSpace):
        raise NotImplementedError

    def __call__(self, ind1: Individual, ind2: Individual, out_space: HyperSpace):
        # Crossover hyperparams only if they have same params
        params_1 = ind1.dna.get_assigned_params()
        params_2 = ind2.dna.get_assigned_params()

        assert len(params_1) == len(params_2)
        for p1, p2 in zip(params_1, params_2):
            assert p1.alias == p2.alias

        out = self.do(ind1, ind2, out_space)
        assert out.all_assigned
        return out


class SinglePointCrossOver(Recombination):

    def do(self, ind1: Individual, ind2: Individual, out_space: HyperSpace):

        params_1 = ind1.dna.get_assigned_params()
        params_2 = ind2.dna.get_assigned_params()
        n_params = len(params_1)
        cut_i = self.random_state.randint(1, n_params - 2)  # ensure offspring has dna from both parents

        # cross over
        for i, hp in enumerate(out_space.params_iterator):
            if i < cut_i:
                # comes from the first parent
                hp.assign(params_1[i].value)
            else:
                hp.assign(params_2[i].value)

        return out_space


class ShuffleCrossOver(Recombination):

    def do(self, ind1: Individual, ind2: Individual, out_space: HyperSpace):
        params_1 = ind1.dna.get_assigned_params()
        params_2 = ind2.dna.get_assigned_params()

        n_params = len(params_1)

        # rearrange dna & single point crossover
        cs_point = self.random_state.randint(1, n_params - 2)
        R = self.random_state.permutation(len(params_1))
        t1_params = []
        t2_params = []
        for i in range(n_params):
            if i < cs_point:
                t1_params[i] = params_1[R[i]]
                t2_params[i] = params_2[R[i]]
            else:
                t1_params[i] = params_2[R[i]]
                t2_params[i] = params_1[R[i]]

        c1_params = []
        c2_params = []
        for i in range(n_params):
            c1_params[R[i]] = c1_params[i]
            c2_params[R[i]] = c2_params[i]

        # select the first child
        for i, hp in enumerate(out_space.params_iterator):
            hp.assign(c1_params[i])

        return out_space


class UniformCrossover(Recombination):
    def __init__(self, random_state=None):
        super().__init__(random_state)
        self.p = 0.5

    def do(self, ind1: Individual, ind2: Individual, out_space: HyperSpace):

        params_1 = ind1.dna.get_assigned_params()
        params_2 = ind2.dna.get_assigned_params()

        # crossover
        for i, hp in enumerate(out_space.params_iterator):
            if self.random_state.random() >= self.p:
                hp.assign(params_1[i].value)
            else:
                hp.assign(params_2[i].value)

        assert out_space.all_assigned
        return out_space


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


class MOEADSearcher(Searcher):
    """
    References
    ----------
        [1]. Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, Dec. 2007, doi: 10.1109/TEVC.2007.892759.
        [2]. Das I, Dennis J E. "Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems[J]." SIAM Journal on Optimization, 1998, 8(3): 631-657.
        [3]. A. Jaszkiewicz, “On the performance of multiple-objective genetic local search on the 0/1 knapsack problem – A comparative experiment,” IEEE Trans. Evol. Comput., vol. 6, no. 4, pp. 402–412, Aug. 2002
    """

    def __init__(self, space_fn, n_objectives, n_sampling=5, n_neighbors=2,
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
        Searcher.__init__(self, space_fn=space_fn, optimize_direction=optimize_direction,
                          use_meta_learner=use_meta_learner, space_sample_validation_fn=space_sample_validation_fn)

        if optimize_direction != OptimizeDirection.Minimize:
            raise ValueError("optimization towards maximization is not supported.")

        self.n_objectives = n_objectives

        self.mutate_probability = mutate_probability

        weight_vectors = self.init_mean_vector_by_NBI(n_sampling, self.n_objectives)  # uniform weighted vectors

        if random_state is None:
            self.random_state = get_random_state()
        else:
            self.random_state = np.random.RandomState(seed=random_state)

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
        self.pop = self.init_population(weight_vectors)

        self._solutions = []
        self._sample_count = 0

    @property
    def population_size(self):
        return len(self.pop)


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
        pop = []
        for i in range(pop_size):
            weight_vector = weight_vectors[i]
            space_sample = self._sample_and_check(self._random_sample)
            pop.append(Individual(space_sample, weight_vector, self.random_state))

        # euler distances matrix
        euler_distances = self.calc_euler_distance(weight_vectors)

        # calc neighbors of each vecto
        for i in range(pop_size):
            sorted_distances = np.argsort(euler_distances[i])
            neighbors = [pop[sorted_distances[i]] for i in range(self.n_neighbors)]
            pop[i].set_neighbors(neighbors)

        return pop

    @staticmethod
    def calc_pf(n_objectives, solutions):
        """Update Pareto front and pareto optimal solution.
          Compare solution with solutions in pareto front, if there are solutions dominated by new solution ,
          then replace them.

        :param dna: new evaluated solution
        :param scores: scores of new solution
        :return:
        """

        def dominate(s1_scores, s2_scores):
            # return: is s1 dominate s2
            ret = []
            for j in range(n_objectives):
                if s1_scores[j] < s2_scores[j]:
                    ret.append(1)
                elif s1_scores[j] == s2_scores[j]:
                    ret.append(0)
                else:
                    return False  # s1 does not dominate s2
            if np.sum(np.array(ret)) >= 1:
                return True  # s1 has at least one metric better that s2
            else:
                return False

        def find_non_dominated_solu(input_solu):
            for solu in solutions:
                if solu == input_solu:
                    continue
                if dominate(solu[1], input_solu[1]):
                    # solu_i has non-dominated solution
                    return solu
            return None  # this is a pareto optimal

        # find non-dominated solution for every solution
        pf_list = list(filter(lambda s: find_non_dominated_solu(s) is None, solutions))

        return pf_list

    def get_pf(self):
        # TODO rename to non-dominated set
        return self.calc_pf(self.n_objectives, self._solutions)

    def single_point_mutate(self, sample_space, out_space, mutate_probability):

        if self.random_state.rand(0, 1) < mutate_probability:
            return sample_space

        # perform mutate
        assert sample_space.all_assigned
        parent_params = sample_space.get_assigned_params()
        pos = self.random_state.randint(0, len(parent_params))
        for i, hp in enumerate(out_space.params_iterator):
            if i > (len(parent_params) - 1) or not parent_params[i].same_config(hp):
                hp.random_sample()
            else:
                if i == pos:
                    new_value = hp.random_sample(assign=False)
                    while new_value == parent_params[i].value:
                        new_value = hp.random_sample(assign=False)
                    hp.assign(new_value)
                else:
                    hp.assign(parent_params[i].value)
        return out_space

    def sample(self):
        # priority evaluation of samples that have not been evaluated
        for indi in self.pop:
            if indi.get_scores() is None:
                return indi.dna

        # select the sub-objective
        individual_inx = self._sample_count % len(self.pop)
        individual = self.pop[individual_inx]

        # select neighbors to  crossover and mutate
        try:
            offspring = self.recombination(*individual.random_neighbors(2), self.space_fn())
            MP = self.mutate_probability
        except Exception as e:
            # sine the length between sample space does not match usually cause no enough neighbors
            logger.info("no enough neighbors, skip mutate")
            offspring = individual.random_neighbors(1)[0].dna
            # Must mutate because of failing crossover
            MP = 1
        final_offspring = self.single_point_mutate(offspring, self.space_fn(), MP)

        individual.set_offspring(final_offspring)

        self._sample_count = self._sample_count + 1

        if final_offspring is not None:
            return self._sample_and_check(lambda: final_offspring)
        else:
            space_sample = self._sample_and_check(self._random_sample)
            return space_sample

    def get_best(self):
        return list(map(lambda s: s[0], self.get_pf()))

    def get_reference_point(self):
        """calculate Z in tchebicheff decomposition
        """
        scores_list = []
        for solu in self._solutions:
            scores = solu[1]
            scores_list.append(scores)
        return np.min(np.array(scores_list), axis=0)

    def get_ideal_point(self):
        non_dominated = self.get_pf()
        return np.min(np.array(list(map(lambda v: v[1], non_dominated))), axis=0)

    def get_nadir_point(self):
        non_dominated = self.get_pf()
        return np.max(np.array(list(map(lambda v: v[1], non_dominated))), axis=0)


    def _update_neighbors(self, indi: Individual, dna, scores):
        for neigh in indi.neighbors:
            wv = neigh.weight_vector
            Z = self.get_reference_point()
            nadir = self.get_nadir_point()
            ideal = self.get_ideal_point()

            if (self.decomposition(neigh.get_scores(), wv, Z, ideal, nadir) - self.decomposition(scores, wv, Z, ideal, nadir)) < 0:
                neigh.update_dna(dna, scores)

    def update_result(self, space, result):
        assert space
        assert result
        self._solutions.append((space, result))

        for indi in self.pop:
            if indi.dna == space:
                indi.set_scores(result)

        for indi in self.pop:
            if indi.get_offspring() == space:
                self._update_neighbors(indi, space, result)

    @property
    def parallelizable(self):
        return False
