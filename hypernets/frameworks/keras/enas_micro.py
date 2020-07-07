# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from .enas_common_ops import *


def enas_micro_search_space(arch='NRNR', input_shape=(28, 28, 1), init_filters=64, node_num=4, hp_dict={}):
    space = HyperSpace()
    with space.as_default():
        input = Input(shape=input_shape)
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
            x = conv_cell(hp_dict, type, cell_no, [node0, node1], filters, node_num, is_reduction)
            node0 = node1
            node1 = x
        space.set_inputs(input)
    return space
