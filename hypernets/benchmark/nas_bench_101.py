# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from ..core.ops import InputChoice, ModuleChoice, HyperInput, Identity, MultipleChoice, Choice
from ..core.search_space import HyperSpace
import numpy as np


class NasBench101():
    def __init__(self, num_nodes, ops=None, input='input',
                 output='output'):
        self.num_nodes = num_nodes
        if ops is None:
            ops = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
        self.ops = ops
        self.num_ops = len(ops)
        self.input = input
        self.output = output

    def get_ops(self, node_no):
        ops = [Identity(name=f'node{node_no}_ops{i}') for i in range(self.num_ops)]
        return ops

    def get_space(self):
        space = HyperSpace()
        with space.as_default():
            input0 = HyperInput()
            inputs = [input0]
            last = input0
            for node_no in range(1, self.num_nodes):
                input_i = Identity(hp_inputs=MultipleChoice(list(range(len(inputs))),
                                                            num_chosen_most=len(inputs),
                                                            num_chosen_least=0))(last)
                if node_no < self.num_nodes - 1:
                    node_i = ModuleChoice(self.get_ops(node_no))
                    node_i(input_i)
                else:
                    node_i = Identity(name=f'output')(input_i)
                last = node_i
                inputs.append(last)
            space.set_inputs(input0)
        return space

    def sample2spec(self, space_sample):
        assert space_sample.all_assigned
        edges_choice = []
        ops_choice = []
        for hp in space_sample.assigned_params_stack:
            if isinstance(hp, MultipleChoice):  #
                edges_choice.append(hp.value)
            elif isinstance(hp, Choice):
                ops_choice.append(hp.value)
            else:
                raise ValueError(f'Unsupported hyper parameter:{hp}')

        assert len(edges_choice) == self.num_nodes - 1
        assert len(ops_choice) == self.num_nodes - 2

        matrix = np.zeros(shape=(self.num_nodes, self.num_nodes), dtype=int)
        col = 1
        for rows in edges_choice:
            for row in rows:
                matrix[row][col] = 1
            col += 1

        ops = []
        ops.append(self.input)
        for op in ops_choice:
            ops.append(self.ops[op])
        ops.append(self.output)

        return matrix, ops
