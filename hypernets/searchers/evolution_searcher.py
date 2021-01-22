# -*- coding:utf-8 -*-
"""

"""
import numpy as np

from ..core.searcher import Searcher, OptimizeDirection
from ..utils import logging

logger = logging.get_logger(__name__)


class Individual(object):
    def __init__(self, space_sample, reward):
        self.space_sample = space_sample
        self.reward = reward

    def mutate(self):
        pass


class Population(object):
    def __init__(self, size=50, optimize_direction=OptimizeDirection.Minimize):
        assert isinstance(size, int)
        assert size > 0
        self.size = size
        self.populations = []
        self.optimize_direction = optimize_direction

    @property
    def initializing(self):
        return not len(self.populations) >= self.size

    @property
    def length(self):
        return len(self.populations)

    def append(self, space_sample, reward):
        individual = Individual(space_sample, reward)
        self.populations.append(individual)

    def sample_best(self, sample_size, random_state=np.random.RandomState()):
        if sample_size > self.length:
            sample_size = self.length
        indices = sorted(random_state.choice(range(self.length), sample_size))
        samples = [self.populations[i] for i in indices]
        best = \
            sorted(samples,
                   key=lambda i: i.reward, reverse=self.optimize_direction in ['max', OptimizeDirection.Maximize])[0]
        return best

    def eliminate(self, num=1, regularized=False):
        eliminates = []
        for i in range(num):
            if self.length <= 0:
                break
            if regularized:
                # eliminate oldest
                eliminates.append(self.populations.pop(0))
            else:
                # eliminate worst
                worst = sorted(self.populations, key=lambda i: i.reward,
                               reverse=self.optimize_direction in ['min', OptimizeDirection.Minimize])[0]
                self.populations.remove(worst)
                eliminates.append(worst)
        return eliminates

    def shuffle(self):
        np.random.shuffle(self.populations)

    def mutate(self, parent_space, offspring_space):
        assert parent_space.all_assigned
        parent_params = parent_space.get_assigned_params()
        pos = np.random.randint(0, len(parent_params))
        for i, hp in enumerate(offspring_space.params_iterator):
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
        return offspring_space


class EvolutionSearcher(Searcher):
    """
    Evolutionary Algorithm

    References
    ----------
        Real, Esteban, et al. "Regularized evolution for image classifier architecture search." Proceedings of the aaai conference on artificial intelligence. Vol. 33. 2019.
    """
    def __init__(self, space_fn, population_size, sample_size, regularized=False,
                 candidates_size=10, optimize_direction=OptimizeDirection.Minimize, use_meta_learner=True,
                 space_sample_validation_fn=None):
        """
        :param space_fn: callable, required
            A search space function which when called returns a `HyperSpace` instance
        :param population_size: int, required
            Size of population
        :param sample_size: int, required
            The number of parent candidates selected in each cycle of evolution
        :param regularized: bool
            (default=False), Whether to enable regularized
        :param candidates_size: int, (default=10)
            The number of samples for the meta-learner to evaluate candidate paths when roll out
        :param optimize_direction: 'min' or 'max', (default='min')
            Whether the search process is approaching the maximum or minimum reward value.
        :param use_meta_learner: bool, (default=True)
            Whether to enable meta leaner. Meta-learner aims to evaluate the performance of unseen samples based on
            previously evaluated samples. It provides a practical solution to accurately estimate a search branch with
            many simulations without involving the actual training.
        :param space_sample_validation_fn: callable or None, (default=None)
            Used to verify the validity of samples from the search space, and can be used to add specific constraint
            rules to the search space to reduce the size of the space.
        """
        Searcher.__init__(self, space_fn=space_fn, optimize_direction=optimize_direction,
                          use_meta_learner=use_meta_learner, space_sample_validation_fn=space_sample_validation_fn)
        self.population = Population(size=population_size, optimize_direction=optimize_direction)
        self.sample_size = sample_size
        self.regularized = regularized
        self.candidate_size = candidates_size

    @property
    def parallelizable(self):
        return True

    def sample(self):
        if self.population.initializing:
            space_sample = self._sample_and_check(self._random_sample)
            return space_sample
        else:
            best = self.population.sample_best(self.sample_size)
            offspring = self._get_offspring(best.space_sample)
            if offspring is not None:
                return offspring
            else:
                self.population.populations.remove(best)
                space_sample = self._sample_and_check(self._random_sample)
                return space_sample

    def _get_offspring(self, space_sample):
        if self.use_meta_learner and self.meta_learner is not None:
            candidates = []
            scores = []
            no = 0
            for i in range(self.candidate_size):
                new_space = self.space_fn()
                try:
                    candidate = self._sample_and_check(lambda: self.population.mutate(space_sample, new_space))
                    candidates.append(candidate)
                    scores.append((no, self.meta_learner.predict(candidate)))
                    no += 1
                except:
                    pass
            if len(candidates) <= 0:
                return None

            topn = sorted(scores,
                          key=lambda s: s[1],
                          reverse=self.optimize_direction in ['max', OptimizeDirection.Maximize])[
                   :int(len(candidates) * 0.3)]
            best = topn[np.random.choice(range(len(topn)))]

            if logger.is_info_enabled():
                logger.info(f'get_offspring scores:{best[1]}, index:{best[0]}')

            return candidates[best[0]]
        else:
            new_space = self.space_fn()
            try:
                candidate = self._sample_and_check(lambda: self.population.mutate(space_sample, new_space))
                return candidate
            except:
                return None

    def update_result(self, space_sample, result):
        if not self.population.initializing:
            self.population.eliminate(regularized=self.regularized)
        self.population.append(space_sample, result)
        if self.use_meta_learner and self.meta_learner is not None:
            assert self.meta_learner is not None
            self.meta_learner.new_sample(space_sample)

    def summary(self):
        summary = '\n'.join(
            [f'vectors:{",".join([str(v) for v in individual.space_sample.vectors])}     reward:{individual.reward} '
             for
             individual in
             self.population.populations])
        return summary
