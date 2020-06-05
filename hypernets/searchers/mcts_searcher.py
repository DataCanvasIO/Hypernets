# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from .mcts_core import *
from ..core.searcher import Searcher, OptimizeDirection


class MCTSSearcher(Searcher):
    def __init__(self, space_fn, policy=None, max_node_space=10, optimize_direction=OptimizeDirection.Minimize):
        if policy is None:
            policy = UCT()
        self.tree = MCTree(space_fn, policy, max_node_space=max_node_space)
        Searcher.__init__(self, space_fn, optimize_direction)
        self.best_nodes = {}

    def sample(self):
        print('Sample')

        space_sample, best_node = self.tree.selection_and_expansion()
        print(f'Sample: {best_node.info()}')

        # count = 0
        # while best_node.is_terminal and best_node.visits > 0:
        #     if count > 1000:
        #         raise RuntimeError('Unable to obtain a valid sample.')
        #     space_sample, best_node = self.tree.selection_and_expansion()
        #     count += 1

        space_sample = self.tree.roll_out(space_sample, best_node)
        self.best_nodes[space_sample.space_id] = best_node
        return space_sample

    def get_best(self):
        raise NotImplementedError

    def update_result(self, space, result):
        best_node = self.best_nodes[space.space_id]
        print(f'Update result: space:{space.space_id}, result:{result}, node:{best_node.info()}')
        self.tree.back_propagation(best_node, result)
        print(f'After back propagation: {best_node.info()}')
        print('\n\n')

    def reset(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
