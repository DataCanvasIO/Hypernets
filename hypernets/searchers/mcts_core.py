# -*- coding:utf-8 -*-
"""

"""
import numpy as np
import math
from collections import OrderedDict
from ..utils.common import generate_id


class MCNode(object):
    def __init__(self, id, name, param_sample, parent=None, tree=None, is_terminal=False):
        self.visits = 0
        self.reward = 0.0

        assert name is not None
        if name != 'ROOT':
            assert param_sample.assigned
        self.name = name
        self.param_sample = param_sample
        self.parent = parent
        if parent is not None:
            assert parent.tree is not None
            self.tree = parent.tree
        else:
            assert tree is not None
            self.tree = tree
        self._is_terminal = is_terminal
        self._children = []
        self.id = id

    @property
    def is_terminal(self):
        return self._is_terminal

    @property
    def is_leaf(self):
        return len(self._children) <= 0

    @property
    def children(self):
        return self._children

    def add_child(self, param_sample):
        self.children.append(MCNode(generate_id(), param_sample.label, param_sample, parent=self))

    def set_parent(self, parent):
        assert self.parent is not None
        self.parent = parent

    def expansion(self, param_space, max_space):
        assert not param_space.assigned
        samples = param_space.expansion(max_space=max_space)
        for param_sample in samples:
            self.add_child(param_sample)


class MCTree(object):
    def __init__(self, space_fn, policy, max_node_space):
        self.space_fn = space_fn
        self.policy = policy
        self.max_node_space = max_node_space
        self.root = MCNode('ROOT', param_sample=None, tree=self)
        self._current_node = self.root

    @property
    def current_node(self):
        return self._current_node

    def is_expanded(self, node):
        raise NotImplementedError

    def selection(self, node):
        raise NotImplementedError

    def path_to_node(self, node):
        nodes = [node]
        while node.parent is not None:
            nodes.insert(0, node.parent)
            node = node.parent
        return nodes

    def expansion(self, node):
        space_sample = self.space_fn()
        nodes = self.path_to_node(node)
        for i, hp in enumerate(space_sample.unassigned_iterator):
            if i < len(nodes):
                hp.assign(nodes[i].value)
            else:
                node.expansion(hp, self.max_node_space)
                break

    def simulation(self, node):
        raise NotImplementedError

    def back_propagation(self, node, reward):
        raise NotImplementedError

    def roll_out(self, node):
        raise NotImplementedError


class Policy(object):
    def selection(self, node):
        raise NotImplementedError

    def back_propagation(self, node, reward):
        raise NotImplementedError


class UCT(Policy):
    def __init__(self, exploration_bonus):
        self.exploration_bonus = exploration_bonus

    def selection(self, node):
        node_log_nt = math.log10(node.visits)
        return node.children[
            np.argmax(
                [(child.reward + self.exploration_bonus * math.sqrt(node_log_nt / child.visits)) for child
                 in
                 node.children])
        ]

    def back_propagation(self, node, reward):
        new_reward = node.reward + (reward - node.reward) / (node.visits + 1)
        return new_reward, node.visits + 1
