# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.searchers.mcts_searcher import *
from hypernets.core.searcher import OptimizeDirection
from hypernets.core.ops import *
from hypernets.core.search_space import *

from hypernets.core.meta_learner import MetaLearner
from hypernets.core.trial import TrialHistory, DiskTrialStore, Trial
from hypernets.tests import test_output_dir

import numpy as np


class Test_MCTS():
    def get_space(self):
        space = HyperSpace()
        with space.as_default():
            p1 = Int(1, 100)
            p2 = Choice(['a', 'b'])
            p3 = Bool()
            p4 = Real(0.0, 1.0)
            id1 = Identity(p1=p1)
            id2 = Identity(p2=p2)(id1)
            id3 = Identity(p3=p3)(id2)
            id4 = Identity(p4=p4)(id3)
        return space

    def test_mctree(self):
        tree = MCTree(self.get_space, policy=UCT(), max_node_space=2)
        space_sample, node = tree.selection_and_expansion()
        assert node.param_sample.label == 'Param_Int_1-1-100-1'
        assert space_sample.all_assigned == False

        tree.back_propagation(node, np.random.uniform(0.1, 0.9))
        assert node.visits == 1
        assert tree.root.visits == 1

        space_sample, node = tree.selection_and_expansion()
        assert node.param_sample.label == 'Param_Int_1-1-100-1'
        assert space_sample.all_assigned == False

        tree.back_propagation(node, np.random.uniform(0.1, 0.9))
        assert node.visits == 1
        assert tree.root.visits == 2

        space_sample, node = tree.selection_and_expansion()
        assert node.param_sample.label == 'Param_Choice_1-[\'a\', \'b\']'
        tree.back_propagation(node, np.random.uniform(0.1, 0.9))
        assert node.visits == 1
        assert tree.root.visits == 3
        assert space_sample.all_assigned == False

        space_sample = tree.roll_out(space_sample, node)
        assert space_sample.all_assigned == True

        space_sample, node = tree.selection_and_expansion()
        assert node.param_sample.label == 'Param_Choice_1-[\'a\', \'b\']'
        tree.back_propagation(node, np.random.uniform(0.1, 0.9))
        assert node.visits == 1
        assert tree.root.visits == 4
        assert space_sample.all_assigned == False

        space_sample = tree.roll_out(space_sample, node)
        assert space_sample.all_assigned == True

        tree_str = str(tree.root)
        assert tree_str

    def test_mcts_searcher(self):
        searcher = MCTSSearcher(self.get_space, max_node_space=10)

        for i in range(1000):
            space_sample = searcher.sample()
            assert space_sample.all_assigned == True
            print(space_sample.params_summary())
            searcher.update_result(space_sample, np.random.uniform(0.1, 0.9))

        assert searcher.tree.root.visits == 1000
        assert len(searcher.nodes_map.items()) == 1000

    def test_mcts_searcher_parallelize(self):
        searcher = MCTSSearcher(self.get_space, max_node_space=10)
        history = TrialHistory(OptimizeDirection.Maximize)
        disk_trial_store = DiskTrialStore(f'{test_output_dir}/trial_store')
        meta_learner = MetaLearner(history, 'test_mcts_searcher_parallelize', disk_trial_store)
        searcher.set_meta_learner(meta_learner)
        trial_no = 1
        running_samples = []
        for i in range(10):
            space_sample = searcher.sample()
            running_samples.append(space_sample)
        assert searcher.tree.root.visits == 10
        for sample in running_samples:
            reward = np.random.uniform(0.1, 0.9)
            trial = Trial(sample, trial_no, reward, 10)
            history.append(trial)
            searcher.update_result(sample, reward)
            trial_no += 1

        assert searcher.tree.root.visits == 10
        assert len(searcher.nodes_map.items()) == 10
        running_samples = []
        for i in range(10):
            space_sample = searcher.sample()
            running_samples.append(space_sample)
        assert searcher.tree.root.visits == 20

        for sample in running_samples:
            reward = np.random.uniform(0.1, 0.9)
            trial = Trial(sample, trial_no, reward, 10)
            history.append(trial)
            searcher.update_result(sample, reward)

        assert searcher.tree.root.visits == 20

    # def test_searcher_with_hp(self):
    #     def get_space():
    #         space = HyperSpace()
    #         with space.as_default():
    #             in1 = Input(shape=(10,))
    #             dense1 = Dense(10, activation=Choice(['relu', 'tanh', None]), use_bias=Bool())(in1)
    #             bn1 = BatchNormalization()(dense1)
    #             dropout1 = Dropout(Choice([0.3, 0.4, 0.5]))(bn1)
    #             output = Dense(2, activation='softmax', use_bias=True)(dropout1)
    #         return space
    #
    #     mcts = MCTSSearcher(get_space)
    #     hk = HyperKeras(mcts, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],
    #                     callbacks=[SummaryCallback()])
    #
    #     x = np.random.randint(0, 10000, size=(100, 10))
    #     y = np.random.randint(0, 2, size=(100), dtype='int')
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
