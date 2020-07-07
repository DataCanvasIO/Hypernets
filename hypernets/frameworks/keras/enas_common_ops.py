# -*- coding:utf-8 -*-
"""

"""

from .layers import Dense, Input, BatchNormalization, Activation, Identity, Add, \
    Conv2D, MaxPooling2D, AveragePooling2D, Flatten, SeparableConv2D, AlignFilters
from ...core.search_space import HyperSpace, Bool, Choice, Dynamic, ModuleSpace
from ...core.ops import Permutation, Sequential, Optional, Repeat, Or, InputChoice
import copy


def sepconv2d_bn(no, name_prefix, kernel_size, filters, strides=(1, 1), x=None):
    relu = Activation(activation='relu', name=f'{name_prefix}relu_{no}_')
    if x is not None and isinstance(x, ModuleSpace):
        relu(x)
    sepconv2d = SeparableConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        name=f'{name_prefix}sepconv2d_{no}'
    )(relu)
    bn = BatchNormalization(name=f'{name_prefix}bn_{no}_')(sepconv2d)
    return bn


def sepconv3x3(name_prefix, filters, strides=(1, 1)):
    name_prefix = name_prefix + 'sepconv3x3_'
    sep1 = sepconv2d_bn(0, name_prefix, kernel_size=(3, 3), filters=filters, strides=strides)
    sep2 = sepconv2d_bn(1, name_prefix, kernel_size=(3, 3), filters=filters, strides=strides, x=sep1)
    return sep2


def sepconv5x5(name_prefix, filters, strides=(1, 1)):
    name_prefix = name_prefix + 'sepconv5x5_'
    sep1 = sepconv2d_bn(0, name_prefix, kernel_size=(5, 5), filters=filters, strides=strides)
    sep2 = sepconv2d_bn(1, name_prefix, kernel_size=(5, 5), filters=filters, strides=strides, x=sep1)
    return sep2


def maxpooling3x3(name_prefix, filters, strides=(1, 1)):
    max = MaxPooling2D(pool_size=(3, 3), strides=strides, padding='same', name=f'{name_prefix}maxpooling3x3_')
    # return max
    align = AlignFilters(filters=filters, name_prefix=name_prefix + 'maxpool3x3_align_')(max)
    return align


def avgpooling3x3(name_prefix, filters, strides=(1, 1)):
    name_prefix = name_prefix + 'avgpooling3x3_'
    avg = AveragePooling2D(pool_size=(3, 3), strides=strides, padding='same', name=f'{name_prefix}avgpooling3x3_')
    # return avg
    align = AlignFilters(filters=filters, name_prefix=name_prefix + 'avgpoo3x3_align_')(avg)
    return align


def identity(name_prefix):
    return Identity(name=f'{name_prefix}identity')


def add(x1, x2, name_prefix):
    return Add(name=f'{name_prefix}add_')([x1, x2])


def conv_op(type, cell_no, node_no, left_or_right, inputs, filters):
    assert isinstance(inputs, list)
    assert all([isinstance(m, ModuleSpace) for m in inputs])
    name_prefix = f'{type}_C{cell_no}_N{node_no}_{left_or_right}_'
    ic1 = InputChoice(inputs, 1)(inputs)
    or1 = Or([sepconv5x5(name_prefix, filters),
              sepconv3x3(name_prefix, filters),
              avgpooling3x3(name_prefix, filters),
              maxpooling3x3(name_prefix, filters),
              identity(name_prefix)])(ic1)
    return or1


def conv_node(type, cell_no, node_no, inputs, filters):
    op_left = conv_op(type, cell_no, node_no, 'L', inputs, filters)
    op_right = conv_op(type, cell_no, node_no, 'R', inputs, filters)
    name_prefix = f'{type}_C{cell_no}_N{node_no}_'
    return add(op_left, op_right, name_prefix)


def conv_cell(type, cell_no, inputs, filters, node_num):
    inputs = copy.copy(inputs)
    for node_no in range(node_num):
        node = conv_node(type, cell_no, node_no, inputs, filters)
        inputs.append(node)


def enas_micro_search_space():
    pass
