# -*- coding:utf-8 -*-
"""

"""

from .layers import Dense, Input, BatchNormalization, Activation, Add, \
    Conv2D, MaxPooling2D, AveragePooling2D, Flatten, SeparableConv2D
from .enas_layers import AlignFilters, SafeConcatenate, SafeAdd, Identity
from ...core.search_space import HyperSpace, Bool, Choice, Dynamic, ModuleSpace
from ...core.ops import Permutation, Sequential, Optional, Repeat, Or, InputChoice, ConnectLooseEnd
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
    name_prefix = name_prefix + 'maxpooling3x3_'
    max = MaxPooling2D(pool_size=(3, 3), strides=strides, padding='same', name=f'{name_prefix}pool_')
    # return max
    align = AlignFilters(filters=filters, name_prefix=name_prefix + 'align_')(max)
    return align


def avgpooling3x3(name_prefix, filters, strides=(1, 1)):
    name_prefix = name_prefix + 'avgpooling3x3_'
    avg = AveragePooling2D(pool_size=(3, 3), strides=strides, padding='same', name=f'{name_prefix}pool_')
    # return avg
    align = AlignFilters(filters=filters, name_prefix=name_prefix + 'align_')(avg)
    return align


def identity(name_prefix):
    return Identity(name=f'{name_prefix}identity')


def add(x1, x2, name_prefix, filters):
    return SafeAdd(filters, name_prefix)([x1, x2])


def conv_op(hp_dict, type, cell_no, node_no, left_or_right, inputs, filters, is_reduction=False):
    assert isinstance(inputs, list)
    assert all([isinstance(m, ModuleSpace) for m in inputs])
    name_prefix = f'{type}_C{cell_no}_N{node_no}_{left_or_right}_'

    input_choice_key = f'{type}_N{node_no}_{left_or_right}_input_choice'
    op_choice_key = f'{type}_N{node_no}_{left_or_right}_op_choice'
    hp_choice = hp_dict.get(input_choice_key)
    ic1 = InputChoice(inputs, 1, hp_choice=hp_choice)(inputs)
    if hp_choice is None:
        hp_dict[input_choice_key] = ic1.hp_choice

    hp_strides = Dynamic(lambda_fn=lambda choice: (2, 2) if is_reduction and choice[0] <= 1 else (1, 1),
                         choice=ic1.hp_choice)

    hp_or = hp_dict.get(op_choice_key)
    or1 = Or([sepconv5x5(name_prefix, filters, strides=hp_strides),
              sepconv3x3(name_prefix, filters, strides=hp_strides),
              avgpooling3x3(name_prefix, filters, strides=hp_strides),
              maxpooling3x3(name_prefix, filters, strides=hp_strides),
              identity(name_prefix)], hp_or=hp_or)(ic1)

    if hp_or is None:
        hp_dict[op_choice_key] = or1.hp_or

    return or1


def conv_node(hp_dict, type, cell_no, node_no, inputs, filters, is_reduction=False):
    op_left = conv_op(hp_dict, type, cell_no, node_no, 'L', inputs, filters, is_reduction)
    op_right = conv_op(hp_dict, type, cell_no, node_no, 'R', inputs, filters, is_reduction)
    name_prefix = f'{type}_C{cell_no}_N{node_no}_'
    return add(op_left, op_right, name_prefix, filters)


def conv_cell(hp_dict, type, cell_no, inputs, filters, node_num, is_reduction=False):
    name_prefix = f'{type}_C{cell_no}_'
    inputs = copy.copy(inputs)
    all_nodes = []
    for node_no in range(node_num):
        node = conv_node(hp_dict, type, cell_no, node_no, inputs, filters, is_reduction)
        inputs.append(node)
        all_nodes.append(node)
    cle = ConnectLooseEnd(all_nodes)(all_nodes)
    concat = SafeConcatenate(filters, name_prefix, name=name_prefix + 'safe_concat_')(cle)
    return concat
