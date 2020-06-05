# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.searchers.mcts_core import *
from hypernets.searchers.mcts_searcher import *
from hypernets.core.searcher import OptimizeDirection
from hypernets.core.ops import *
from hypernets.core.search_space import *
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
        assert node.param_sample.label == 'Param_Int_1-1-100'
        assert space_sample.all_assigned == False

        tree.back_propagation(node, np.random.uniform(0.1, 0.9))
        assert node.visits == 1
        assert tree.root.visits == 1

        space_sample, node = tree.selection_and_expansion()
        assert node.param_sample.label == 'Param_Int_1-1-100'
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

    def test_mcts_searcher(self):
        searcher = MCTSSearcher(self.get_space, max_node_space=10)

        for i in range(100):
            space_sample = searcher.sample()
            assert space_sample.all_assigned == True
            space_sample.params_summary()
            searcher.update_result(space_sample, np.random.uniform(0.1, 0.9))

        assert searcher.tree.root.visits == 100
        assert len(searcher.best_nodes.items()) == 100
