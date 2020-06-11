# -*- coding:utf-8 -*-
"""

"""
from ..core.searcher import Searcher, OptimizeDirection
from ..core.meta_learner import MetaLearner
import numpy as np


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
        self.initializing = True

    @property
    def length(self):
        return len(self.populations)

    def append(self, space_sample, reward):
        individual = Individual(space_sample, reward)
        self.populations.append(individual)
        if len(self.populations) >= self.size:
            self.initializing = False

    def sample_best(self, sample_size, random_state=np.random.RandomState()):
        if sample_size > self.length:
            sample_size = self.length
        indices = sorted(random_state.choice(range(self.length), sample_size))
        samples = [self.populations[i] for i in indices]
        best = \
            sorted(samples,
                   key=lambda i: i.reward if self.optimize_direction == OptimizeDirection.Minimize else -i.reward)[
                0]
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
                worst = sorted(self.populations, key=lambda
                    i: -i.reward if self.optimize_direction == OptimizeDirection.Minimize else i.reward)[
                    0]
                self.populations.remove(worst)
                eliminates.append(worst)
        return eliminates

    def shuffle(self):
        np.random.shuffle(self.populations)

    def mutate(self, parent_space, offspring_space):
        parent_params = parent_space.get_assignable_params()
        pos = np.random.randint(0, len(parent_params))
        for i, hp in enumerate(offspring_space.unassigned_iterator):
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
    def __init__(self, space_fn, population_size, sample_size, regularized=False, use_meta_learner=True,
                 candidates_size=10,
                 optimize_direction=OptimizeDirection.Minimize, ):
        Searcher.__init__(self, space_fn=space_fn, optimize_direction=optimize_direction)
        self.population = Population(size=population_size, optimize_direction=optimize_direction)
        self.sample_size = sample_size
        self.use_meta_learner = use_meta_learner
        self.regularized = regularized
        self.histroy = None
        self.meta_learner = None
        self.candidate_size = candidates_size

    def sample(self, history):
        if self.use_meta_learner and history is not None and self.histroy is None:
            self.histroy = history
            self.meta_learner = MetaLearner(history)
        if self.population.initializing:
            space_sample = self.space_fn()
            space_sample.random_sample()
            return space_sample
        else:
            best = self.population.sample_best(self.sample_size)
            offspring = self._get_offspring(best.space_sample)
            return offspring

    def _get_offspring(self, space_sample):
        if self.use_meta_learner and self.meta_learner is not None:
            candidates = []
            scores = []
            for i in range(self.candidate_size):
                new_space = self.space_fn()
                candidate = self.population.mutate(space_sample, new_space)
                candidates.append(candidate)
                scores.append(self.meta_learner.predict(candidate))
            index = np.argmax(scores)
            print(f'get_offspring scores:{scores}, argmax:{index}')
            return candidates[index]
        else:
            new_space = self.space_fn()
            candidate = self.population.mutate(space_sample, new_space)
            return candidate

    def update_result(self, space_sample, result):
        if not self.population.initializing:
            self.population.eliminate(regularized=self.regularized)
        self.population.append(space_sample, result)
        if self.use_meta_learner and self.meta_learner is not None:
            assert self.meta_learner is not None
            self.meta_learner.new_sample(space_sample)
