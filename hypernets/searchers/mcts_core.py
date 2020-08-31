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
        self.rewards = []
        self.simulation_rewards = []

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
    def expanded(self):
        return all([n.visits > 0 for n in self.children])

    @property
    def is_leaf(self):
        return len(self._children) <= 0

    @property
    def children(self):
        return self._children

    @property
    def depth(self):
        d = 0
        node = self.parent
        while node is not None:
            d += 1
            node = node.parent
        return d

    def add_child(self, param_sample):
        self.children.append(MCNode(generate_id(), param_sample.label, param_sample, parent=self))

    def set_parent(self, parent):
        assert self.parent is not None
        self.parent = parent

    def expansion(self, param_space, max_space):
        #print(f'Node expanssion: param_space:{param_space.label}')
        assert not param_space.assigned
        samples = param_space.expansion(sample_num=max_space)
        for param_sample in samples:
            self.add_child(param_sample)

    def random_sample(self):
        child = np.random.choice(self._children)
        return child

    def info(self):
        return f'Reward:{self.reward}, Visits:{self.visits}, Name:{self.name}, Alias:{self.param_sample.alias if self.param_sample is not None else None}, value:{self.param_sample.value if self.param_sample is not None else None}, depth:{self.depth},  children:{len(self._children)}, is_terminal:{self.is_terminal}'

    def __str__(self, level=0):
        indent = "\t" * level

        str = f'{indent}{self.info()}\n'
        for child in self.children:
            str += child.__str__(level + 1)
        return str


class MCTree(object):
    def __init__(self, space_fn, policy, max_node_space):
        self.space_fn = space_fn
        self.policy = policy
        self.max_node_space = max_node_space
        self.root = MCNode(generate_id(), 'ROOT', param_sample=None, tree=self)
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
                #print(f'Tree selection:{node.info()}')
                node = self.policy.selection(node)
                if node.visits <= 0:
                    break

        space_sample = self.node_to_space(node)
        return space_sample, node

    def path_to_node(self, node):
        nodes = []
        while node.parent is not None:
            nodes.insert(0, node)
            node = node.parent
        return nodes

    def node_to_space(self, node):
        space_sample = self.space_fn()
        nodes = self.path_to_node(node)
        i = 0
        for hp in space_sample.params_iterator:
            if i >= len(nodes):
                break
            hp.assign(nodes[i].param_sample.value)
            i += 1
        return space_sample

    def expansion(self, node):
        #print(f'Tree expansion:{node.info()}')
        space_sample = self.space_fn()
        nodes = self.path_to_node(node)
        i = 0
        for hp in space_sample.params_iterator:
            if i < len(nodes):
                hp.assign(nodes[i].param_sample.value)
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

    def back_propagation(self, node, reward, is_simulation=False):
        while node is not None:
            node.reward, node.visits = self.policy.back_propagation(node, reward, is_simulation)
            node = node.parent

    def roll_out(self, space_sample, node):
        #print(f'Tree roll out:{node.info()}')
        terminal = True
        for hp in space_sample.params_iterator:
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
    def __init__(self, exploration_bonus=0.6):
        self.exploration_bonus = exploration_bonus

    def selection(self, node):
        node_log_nt = math.log10(node.visits)
        scores = [(child.reward + self.exploration_bonus * math.sqrt(
            node_log_nt / child.visits)) if child.visits > 0 else np.inf for child
                  in
                  node.children]
        details = [(child.param_sample.value, np.sum(child.rewards), child.reward, len(child.rewards)) for child in
                   node.children]
        index = np.argmax(scores)
        selected = node.children[index]
        #print(f'UCT selection: scores:{scores}, index:{index}, selected:{selected.info()}')
        #print(f'Detials: {details}')
        #print('*******************************************************************************')

        return selected

    def back_propagation(self, node, reward, is_simulation=False):
        if is_simulation:
            node.simulation_rewards.append(reward)
            # return np.average(node.rewards + [reward]), len(node.rewards) + 1
        else:
            node.rewards.append(reward)
            # return np.average(node.rewards), len(node.rewards)
        avg_reward = np.average(node.rewards) if len(node.rewards) > 0 else 0.0
        if len(node.simulation_rewards) > 0:
            avg_sim_reward = np.average(node.simulation_rewards)
            new_reward = (avg_reward + avg_sim_reward) / 2
            visits = len(node.simulation_rewards)
        else:
            new_reward = avg_reward
            visits = len(node.rewards)
        return new_reward, visits
