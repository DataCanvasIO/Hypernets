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

    def set_terminal(self):
        self._is_terminal = True
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

    def random_sample(self):
        child = np.random.choice(self._children)
        return child


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

    def selection_and_expansion(self):
        node = self.root
        while not node.is_terminal:
            if node.is_leaf:
                space_sample, child = self.expansion(node)
                if child != node:
                    return space_sample, child
            else:
                node = self.policy.selection(node)

        return None, node

    def path_to_node(self, node):
        nodes = [node]
        while node.parent is not None:
            nodes.insert(0, node.parent)
            node = node.parent
        return nodes

    def node_to_space(self, node):
        assert node.is_terminal

        space_sample = self.space_fn()
        nodes = self.path_to_node(node)
        i = 0
        for hp in space_sample.unassigned_iterator:
            assert i < len(nodes)
            hp.assign(nodes[i].value)
            i += 1
        return space_sample

    def expansion(self, node):
        space_sample = self.space_fn()
        nodes = self.path_to_node(node)
        i = 0
        for hp in space_sample.unassigned_iterator:
            if i < len(nodes):
                hp.assign(nodes[i].value)
            else:
                node.expansion(hp, self.max_node_space)
                child = node.random_sample()
                hp.assign(child.param_sample.value)
                return space_sample, child
            i += 1
        node.set_terminal()
        return space_sample, node

    def simulation(self, node):
        raise NotImplementedError

    def back_propagation(self, node, reward):
        while node.parent is not None:
            self.policy.back_propagation(node, reward)
            node = node.parent

    def roll_out(self, space_sample, node):
        terminal = True
        for hp in space_sample.unassigned_iterator:
            terminal = False
            hp.random_sample()
        if terminal:
            node.set_terminal()
        return space_sample


class BasePolicy(object):
    def selection(self, node):
        raise NotImplementedError

    def back_propagation(self, node, reward):
        raise NotImplementedError


class UCT(BasePolicy):
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
