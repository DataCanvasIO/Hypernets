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
    def test_mctree(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                p1 = Int(1, 100)
                p2 = Choice(['a', 'b', 'c', 'd'])
                p3 = Bool()
                p4 = Real(0.0, 1.0)
                id1 = Identity(p1=p1)
                id2 = Identity(p2=p2)(id1)
                id3 = Identity(p3=p3)(id2)
                id4 = Identity(p4=p4)(id3)
            return space

        tree = MCTree(get_space, policy=UCT(), max_node_space=10)
        space_sample, node = tree.selection_and_expansion()
        assert node.param_sample.label == 'Param_Int_1-1-100'

        tree.back_propagation(node, 0.9)
        assert node.visits == 1

        space_sample, node = tree.selection_and_expansion()
        assert node.param_sample.label == 'Param_Choice_1-[\'a\', \'b\', \'c\', \'d\']'
        tree.back_propagation(node, 0.9)
        assert node.visits == 1
        assert tree.root.visits == 2
