# -*- coding:utf-8 -*-
"""

"""
from ..core.searcher import Searcher, OptimizeDirection
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
        assert parent_space.all_assigned
        parent_params = parent_space.get_assigned_params()
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
    def __init__(self, space_fn, population_size, sample_size, regularized=False,
                 candidates_size=10, optimize_direction=OptimizeDirection.Minimize, use_meta_learner=True):
        Searcher.__init__(self, space_fn=space_fn, optimize_direction=optimize_direction,
                          use_meta_learner=use_meta_learner)
        self.population = Population(size=population_size, optimize_direction=optimize_direction)
        self.sample_size = sample_size
        self.regularized = regularized
        self.candidate_size = candidates_size

    def sample(self):
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
                scores.append((i, self.meta_learner.predict(candidate)))
            topn = sorted(scores,
                          key=lambda s: s[1] if self.optimize_direction == OptimizeDirection.Minimize else -s[1])[
                   :int(self.candidate_size * 0.3)]
            best = topn[np.random.choice(range(len(topn)))]
            print(f'get_offspring scores:{best[1]}, index:{best[0]}')
            return candidates[best[0]]
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

    def summary(self):
        summary = '\n'.join(
            [f'vectors:{",".join([str(v) for v in individual.space_sample.vectors])}     reward:{individual.reward} '
             for
             individual in
             self.population.populations])
        return summary
