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
        self._son = None
        self._scores = None

    def get_son(self):
        return self._son

    def set_son(self, son):
        self._son = son

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
        """Random select neighbors
        :param n:
        :param same_param: ensure neighbors has same params
        :return:
        """
        neighbor_len = len(self.neighbors)
        if n > neighbor_len:
            raise RuntimeError(f"required neighbors = {n} bigger that all neighbors = {neighbor_len} .")

        if n == 1:
            return [self.neighbors[self.random_state.randint(0, neighbor_len, size=n)[0]]]

        # group by params
        params_list = []
        for neighbor in self.neighbors:
            assert neighbor.dna.all_assigned
            # TODO use hp.same_config
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


class MOEADSearcher(Searcher):
    """
    References
    ----------
        [1]. Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, Dec. 2007, doi: 10.1109/TEVC.2007.892759.
        [2]. Das I, Dennis J E. "Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems[J]." SIAM Journal on Optimization, 1998, 8(3): 631-657.
        [3]. A. Jaszkiewicz, “On the performance of multiple-objective genetic local search on the 0/1 knapsack problem – A comparative experiment,” IEEE Trans. Evol. Comput., vol. 6, no. 4, pp. 402–412, Aug. 2002
    """

    def __init__(self, space_fn, objectives, n_sampling=5, n_neighbors=2, mutate_probability=0.3,
                 decomposition=None, pbi_theta=0.5, optimize_direction=OptimizeDirection.Minimize, use_meta_learner=False,
                 space_sample_validation_fn=None, random_state=None):
        """
        :param space_fn: 
        :param n_sampling: the number of samples in each objective
        :param objectives: name of objectives
        :param n_neighbors: num of neighbors to mating
        :param mutate_probability:
        :param decomposition: decomposition approch, default is None one of tchebicheff,weighted_sum
        :param optimize_direction:
        :param use_meta_learner:
        :param space_sample_validation_fn:
        :param random_state:
        """
        Searcher.__init__(self, space_fn=space_fn, optimize_direction=optimize_direction,
                          use_meta_learner=use_meta_learner, space_sample_validation_fn=space_sample_validation_fn)

        self.objectives = objectives

        self.mutate_probability = mutate_probability

        self.pbi_theta = pbi_theta

        weight_vectors = self.init_mean_vector_by_NBI(n_sampling, self.n_objectives)  # uniform weighted vectors

        n_vectors = weight_vectors.shape[0]
        if n_neighbors > n_vectors:
            raise RuntimeError(f"n_neighbors should less that {n_vectors - 1}")

        if decomposition is None:
            self.decomposition = self.tchebicheff_decomposition
        else:
            mapping = {'tchebicheff': self.tchebicheff_decomposition,
                       'weighted_sum': self.weighted_sum_decomposition}
            if decomposition in mapping:
                self.decomposition = mapping[decomposition]
            else:
                raise RuntimeError(f'unseen decomposition apporch {decomposition}.')

        if random_state is None:
            self.random_state = get_random_state()
        else:
            self.random_state = np.random.RandomState(seed=random_state)

        self.n_neighbors = n_neighbors
        self.pop = self.init_population(weight_vectors)

        self._solutions = []
        self._sample_count = 0

    @property
    def population_size(self):
        return len(self.pop)

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

    def get_pf(self):
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
            for j, o_name in enumerate(self.objectives):
                if s1_scores[o_name] < s2_scores[o_name]:
                    ret.append(1)
                elif s1_scores[o_name] == s2_scores[o_name]:
                    ret.append(0)
                else:
                    return False  # s1 does not dominate s2
            if np.sum(np.array(ret)) >= 1:
                return True  # s1 has at least one metric better that s2
            else:
                return False

        def find_non_dominated_solu(input_solu):
            for solu in self._solutions:
                if solu == input_solu:
                    continue
                if dominate(solu[1], input_solu[1]):
                    # solu_i has non-dominated solution
                    return solu
            return None  # this is a pareto optimal

        # find non-dominated solution for every solution
        pf_list = list(filter(lambda s: find_non_dominated_solu(s) is None, self._solutions))

        return pf_list

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

    def single_point_crossover(self, ind1: Individual, ind2: Individual, out_space: HyperSpace):
        """Crossover hyperparams only if they have same params
        """
        params_1 = ind1.dna.get_assigned_params()
        params_2 = ind2.dna.get_assigned_params()

        assert len(params_1) == len(params_2)
        for p1, p2 in zip(params_1, params_2):
            assert p1.alias == p2.alias

        # random crossover point
        if len(params_1) != len(params_2):
            print(params_1)
            print(params_2)

        assert len(params_1) == len(params_2)
        n_params = len(params_1)
        cut_i = self.random_state.randint(1, n_params - 2)  # ensure offspring has dna from both parents

        # cross over
        for i, hp in enumerate(out_space.params_iterator):
            if i < cut_i:
                # comes from the first parent
                hp.assign(params_1[i].value)
            else:
                hp.assign(params_2[i].value)

        assert out_space.all_assigned

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
            offspring = self.single_point_crossover(*individual.random_neighbors(2), self.space_fn())
            MP = self.mutate_probability
        except Exception as e:
            # sine the length between sample space does not match usually cause no enough neighbors
            logger.info("no enough neighbors, skip mutate")
            offspring = individual.random_neighbors(1)[0].dna
            # Must mutate because of failing crossover
            MP = 1
        final_offspring = self.single_point_mutate(offspring, self.space_fn(), MP)

        individual.set_son(final_offspring)

        self._sample_count = self._sample_count + 1

        if final_offspring is not None:
            return self._sample_and_check(lambda: final_offspring)
        else:
            space_sample = self._sample_and_check(self._random_sample)
            return space_sample

    def get_best(self):
        return list(map(lambda s: s[0], self.get_pf()))

    def compare_scores(self, scores1: dict, scores2: dict,
                       weight_vector, de_func):
        kwargs = {}
        if de_func == self.pbi_decomposition:
            kwargs['theta'] = self.pbi_theta
        return de_func(scores2, weight_vector, **kwargs) - de_func(scores1, weight_vector, **kwargs)

    def _to_score_vec(self, scores: dict):
        return np.array([scores[k] for k in self.objectives])

    def weighted_sum_decomposition(self, scores: dict, weight_vector):
        return (self._to_score_vec(scores) * weight_vector).sum()

    def tchebicheff_decomposition(self, scores, weight_vector):
        Z = self.get_reference_point()
        F = self._to_score_vec(scores)
        return np.max(np.abs(F - Z) * weight_vector)

    def pbi_decomposition(self, scores: dict, weight_vector: np.ndarray, theta: float = 0.5):
        """An implementation of "Boundary Intersection Approach base on penalty"
        :param scores
        :param weight_vector
        :param theta: Penalty F deviates from the weight vector.
        """
        F = self._to_score_vec(scores)
        Z = self.get_reference_point()  # virtual ZERO point
        d1 = ((Z - F) * weight_vector) / np.linalg.norm(weight_vector)
        d2 = np.linalg.norm((Z - weight_vector * d1) - F)

        return d1 + d2 * theta

    def get_reference_point(self):
        """calculate Z in tchebicheff decomposition
        """
        scores_list = []
        for solu in self._solutions:
            scores = solu[1]
            scores_list.append([scores[o_name] for o_name in self.objectives])

        return np.min(np.array(scores_list), axis=0)

    def _update_neighbors(self, indi: Individual, dna, scores):
        for neigh in indi.neighbors:
            if self.compare_scores(neigh.get_scores(),
                                   scores, neigh.weight_vector, self.decomposition) < 0:
                neigh.update_dna(dna, scores)

    def update_result(self, space, result):
        assert space
        assert result
        self._solutions.append((space, result))

        for indi in self.pop:
            if indi.dna == space:
                indi.set_scores(result)

        for indi in self.pop:
            if indi.get_son() == space:
                self._update_neighbors(indi, space, result)

    @property
    def parallelizable(self):
        return False
