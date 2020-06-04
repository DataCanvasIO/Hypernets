# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from .mcts_core import *
from ..core.searcher import Searcher, OptimizeDirection


class MCTSSearcher(Searcher):
    def __init__(self, space_fn, policy=None, optimize_direction=OptimizeDirection.Minimize):
        if policy is None:
            policy = UCT()
        self.tree = MCTree(space_fn, policy)
        Searcher.__init__(self, space_fn, optimize_direction)
        self.best_nodes = {}

    def sample(self):
        space_sample, best_node = self.tree.selection_and_expansion()
        if space_sample is None:
            space_sample = self.tree.node_to_space(best_node)
        else:
            space_sample = self.tree.roll_out(space_sample, best_node)
        self.best_nodes[space_sample.id] = best_node
        return space_sample

    def get_best(self):
        raise NotImplementedError

    def update_result(self, space, result):
        best_node = self.best_nodes[space.id]
        self.tree.back_propagation(best_node, result)

    def reset(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
