# -*- coding:utf-8 -*-
"""

"""
import numpy as np

from hypernets.core.ops import Identity
from hypernets.core.search_space import HyperSpace, Int, Real, Choice, Bool
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.evolution_searcher import Population


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

        # population = Population(optimize_direction=OptimizeDirection.Maximize)
        population = Population(optimize_direction='max')

        space1 = get_space()
        space1.random_sample()
        assert space1.all_assigned

        space2 = get_space()
        assert not space2.all_assigned
        new_space = population.mutate(space1, space2)

        pv1 = list(space1.get_assigned_param_values().values())
        pv2 = list(space2.get_assigned_param_values().values())

        assert space2.all_assigned
        assert new_space.all_assigned
        assert np.sum([v1 != v2 for v1, v2 in zip(pv1, pv2)]) == 1

    # def test_searcher_with_hp(self):
    #     def get_space():
    #         space = HyperSpace()
    #         with space.as_default():
    #             in1 = Input(shape=(10,))
    #             in2 = Input(shape=(20,))
    #             in3 = Input(shape=(1,))
    #             concat = Concatenate()([in1, in2, in3])
    #             dense1 = Dense(10, activation=Choice(['relu', 'tanh', None]), use_bias=Bool())(concat)
    #             bn1 = BatchNormalization()(dense1)
    #             dropout1 = Dropout(Choice([0.3, 0.4, 0.5]))(bn1)
    #             output = Dense(2, activation='softmax', use_bias=True)(dropout1)
    #         return space
    #
    #     rs = EvolutionSearcher(get_space, 5, 3, regularized=False, optimize_direction=OptimizeDirection.Maximize)
    #     hk = HyperKeras(rs, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],
    #                     callbacks=[SummaryCallback()])
    #
    #     x1 = np.random.randint(0, 10000, size=(100, 10))
    #     x2 = np.random.randint(0, 100, size=(100, 20))
    #     x3 = np.random.normal(1.0, 100.0, size=(100))
    #     y = np.random.randint(0, 2, size=(100), dtype='int')
    #     x = [x1, x2, x3]
    #
    #     hk.search(x, y, x, y, max_trials=10)
    #     assert hk.get_best_trial()
    #     best_trial = hk.get_best_trial()
    #
    #     estimator = hk.final_train(best_trial.space_sample, x, y)
    #     score = estimator.predict(x)
    #     result = estimator.evaluate(x, y)
    #     assert len(score) == 100
    #     assert result
