# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from .enas_common_ops import *
from .layers import Input
from .enas_layers import FactorizedReduction
from hypernets.core.search_space import HyperSpace


def enas_micro_search_space(arch='NRNR', input_shape=(28, 28, 1), init_filters=64, node_num=4, data_format=None,
                            hp_dict={}):
    space = HyperSpace()
    with space.as_default():
        input = Input(shape=input_shape, name='0_input')
        node0 = input
        node1 = input
        reduction_no = 0
        normal_no = 0

        for l in arch:
            if l == 'N':
                normal_no += 1
                type = 'normal'
                cell_no = normal_no
                is_reduction = False
            else:
                reduction_no += 1
                type = 'reduction'
                cell_no = reduction_no
                is_reduction = True
            filters = (2 ** reduction_no) * init_filters

            if is_reduction:
                node0 = node1
                node1 = FactorizedReduction(filters, f'{type}_C{cell_no}_', data_format)(node1)
            x = conv_layer(hp_dict, f'{normal_no + reduction_no}_{type}', cell_no, [node0, node1], filters, node_num,
                           is_reduction)
            node0 = node1
            node1 = x
        space.set_inputs(input)
    return space
