# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from .mcts_core import *
from ..core.searcher import Searcher, OptimizeDirection


class MCTSSearcher(Searcher):
    def __init__(self, space_fn, policy, optimize_direction=OptimizeDirection.Minimize):
        self.tree = MCTree(space_fn, policy)
        Searcher.__init__(self, space_fn, optimize_direction)

    def sample(self):
        if self.tree.current_node == self.tree.root:
            pass
            # expansion
        elif self.tree.current_node.is_terminal:
            pass

        best_node = self.tree.selection()

        pass

    def get_best(self):
        raise NotImplementedError

    def update_result(self, space, result):
        pass

    def reset(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
