# -*- coding:utf-8 -*-
"""

"""
from hypernets.searchers.evolution_searcher import *
from hypernets.core.searcher import OptimizeDirection
from hypernets.core.search_space import *
from hypernets.core.ops import *
import numpy as np


class Test_Evolution():
    def test_population(self):
        population = Population(optimize_direction=OptimizeDirection.Maximize)
        population.append('a', 0)
        population.append('b', 1)
        population.append('c', 2)
        population.append('d', 3)
        population.append('e', 4)
        population.append('f', 5)
        population.append('g', 6)
        population.append('h', 7)
        population.append('i', 8)
        population.append('i', 9)

        b1 = population.sample_best(25, np.random.RandomState(9527))
        assert b1.reward == 8

        population = Population(optimize_direction=OptimizeDirection.Minimize)
        population.append('a', 0)
        population.append('b', 1)
        population.append('c', 2)
        population.append('d', 3)
        population.append('e', 4)
        population.append('f', 5)
        population.append('g', 6)
        population.append('h', 7)
        population.append('i', 8)
        population.append('i', 9)

        b2 = population.sample_best(25, np.random.RandomState(9527))
        assert b2.reward == 0

    def test_eliminate(self):
        population = Population(optimize_direction=OptimizeDirection.Maximize)
        population.append('a', 4)
        population.append('b', 3)
        population.append('c', 2)
        population.append('d', 1)
        population.append('e', 0)
        population.append('f', 5)
        population.append('g', 6)
        population.append('h', 7)
        population.append('i', 8)
        population.append('j', 9)

        eliminates = population.eliminate(2, regularized=True)
        assert eliminates[0].space_sample == 'a' and eliminates[1].space_sample == 'b'

        eliminates = population.eliminate(2, regularized=False)
        assert eliminates[0].space_sample == 'e' and eliminates[1].space_sample == 'd'

    def test_mutate(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                id1 = Identity(p1=Int(0, 10), p2=Choice(['a', 'b']))
                id2 = Identity(p3=Real(0., 1.), p4=Bool())(id1)
            return space

        population = Population(optimize_direction=OptimizeDirection.Maximize)

        space1 = get_space()
        space1.random_sample()
        assert space1.all_assigned

        space2 = get_space()
        assert not space2.all_assigned
        new_space = population.mutate(space2, space1)

        pv1 = list(space1.get_assignable_param_values().values())
        pv2 = list(space2.get_assignable_param_values().values())

        assert space2.all_assigned
        assert new_space.all_assigned
        assert np.sum([v1 != v2 for v1, v2 in zip(pv1, pv2)]) == 1
