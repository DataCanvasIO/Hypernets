# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from .enas_common_ops import *
from .layers import Input
from .enas_layers import FactorizedReduction
from hypernets.core.search_space import HyperSpace


def enas_micro_search_space(arch='NRNR', input_shape=(28, 28, 1), init_filters=64, node_num=4, data_format=None,
                            classes=10, classification_dropout=0,
                            hp_dict={}, use_input_placeholder=True,
                            weights_cache=None):
    space = HyperSpace()
    with space.as_default():
        if use_input_placeholder:
            input = Input(shape=input_shape, name='0_input')
        else:
            input = None
        stem, input = stem_op(input, init_filters, data_format)
        node0 = stem
        node1 = stem
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
                node0 = FactorizedReduction(filters, f'{normal_no + reduction_no}_{type}_C{cell_no}_0', data_format)(
                    node0)
                node1 = FactorizedReduction(filters, f'{normal_no + reduction_no}_{type}_C{cell_no}_1', data_format)(
                    node1)
            x = conv_layer(hp_dict, f'{normal_no + reduction_no}_{type}', cell_no, [node0, node1], filters, node_num,
                           is_reduction)
            node0 = node1
            node1 = x
        logit = classification(x, classes, classification_dropout, data_format)
        space.set_inputs(input)
        if weights_cache is not None:
            space.weights_cache = weights_cache

    return space
