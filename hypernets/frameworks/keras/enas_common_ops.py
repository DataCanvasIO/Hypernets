# -*- coding:utf-8 -*-
"""

"""

from hypernets.frameworks.keras.enas_layers import SafeMerge, Identity, CalibrateSize
from hypernets.frameworks.keras.layers import BatchNormalization, Activation, Add, MaxPooling2D, AveragePooling2D, \
    SeparableConv2D, Conv2D, GlobalAveragePooling2D, Dense, Dropout
from hypernets.core.ops import ModuleChoice, InputChoice, ConnectLooseEnd
from hypernets.core.search_space import ModuleSpace, MultipleChoice, Choice
from tensorflow.python.keras import utils


def sepconv2d_bn(no, name_prefix, kernel_size, filters, strides=(1, 1), data_format=None, x=None):
    relu = Activation(activation='relu', name=f'{name_prefix}relu_{no}_')
    if x is not None and isinstance(x, ModuleSpace):
        relu(x)
    sepconv2d = SeparableConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        data_format=data_format,
        name=f'{name_prefix}sepconv2d_{no}'
    )(relu)
    bn = BatchNormalization(name=f'{name_prefix}bn_{no}_')(sepconv2d)
    return bn


def sepconv3x3(name_prefix, filters, strides=(1, 1), data_format=None):
    name_prefix = name_prefix + 'sepconv3x3_'
    sep1 = sepconv2d_bn(0, name_prefix, kernel_size=(3, 3), filters=filters, strides=strides, data_format=data_format)
    sep2 = sepconv2d_bn(1, name_prefix, kernel_size=(3, 3), filters=filters, strides=strides, data_format=data_format,
                        x=sep1)
    return sep2


def sepconv5x5(name_prefix, filters, strides=(1, 1), data_format=None):
    name_prefix = name_prefix + 'sepconv5x5_'
    sep1 = sepconv2d_bn(0, name_prefix, kernel_size=(5, 5), filters=filters, strides=strides, data_format=data_format)
    sep2 = sepconv2d_bn(1, name_prefix, kernel_size=(5, 5), filters=filters, strides=strides, data_format=data_format,
                        x=sep1)
    return sep2


def maxpooling3x3(name_prefix, filters, strides=(1, 1), data_format=None):
    name_prefix = name_prefix + 'maxpooling3x3_'
    max = MaxPooling2D(pool_size=(3, 3), strides=strides, padding='same', data_format=data_format,
                       name=f'{name_prefix}pool_')
    return max


def avgpooling3x3(name_prefix, filters, strides=(1, 1), data_format=None):
    name_prefix = name_prefix + 'avgpooling3x3_'
    avg = AveragePooling2D(pool_size=(3, 3), strides=strides, padding='same', data_format=data_format,
                           name=f'{name_prefix}pool_')
    return avg


def identity(name_prefix):
    return Identity(name=f'{name_prefix}identity')


def add(name_prefix, inputs):
    return Add(name=f'{name_prefix}add_')(inputs)


def conv_cell(hp_dict, type, cell_no, node_no, left_or_right, inputs, filters, is_reduction=False, data_format=None):
    assert isinstance(inputs, list)
    assert all([isinstance(m, ModuleSpace) for m in inputs])
    name_prefix = f'{type}_C{cell_no}_N{node_no}_{left_or_right}_'

    input_choice_key = f'{type[2:]}_N{node_no}_{left_or_right}_input_choice'
    op_choice_key = f'{type[2:]}_N{node_no}_{left_or_right}_op_choice'
    hp_choice = hp_dict.get(input_choice_key)
    if hp_choice is None:
        hp_choice = MultipleChoice(list(range(len(inputs))), 1, name=input_choice_key)
        hp_dict[input_choice_key] = hp_choice
    ic1 = InputChoice(inputs, 1, hp_choice=hp_choice)(inputs)
    if hp_choice is None:
        hp_dict[input_choice_key] = ic1.hp_choice

    # hp_strides = Dynamic(lambda_fn=lambda choice: (2, 2) if is_reduction and choice[0] <= 1 else (1, 1),
    #                      choice=ic1.hp_choice)
    hp_strides = (1, 1)
    hp_op_choice = hp_dict.get(op_choice_key)
    module_candidates = [sepconv5x5(name_prefix, filters, strides=hp_strides, data_format=data_format),
                         sepconv3x3(name_prefix, filters, strides=hp_strides, data_format=data_format),
                         avgpooling3x3(name_prefix, filters, strides=hp_strides, data_format=data_format),
                         maxpooling3x3(name_prefix, filters, strides=hp_strides, data_format=data_format),
                         identity(name_prefix)]
    if hp_op_choice is None:
        hp_op_choice = Choice(list(range(len(module_candidates))), name=op_choice_key)
        hp_dict[op_choice_key] = hp_op_choice
    op_choice = ModuleChoice(module_candidates, hp_or=hp_op_choice)(ic1)

    return op_choice


def conv_node(hp_dict, type, cell_no, node_no, inputs, filters, is_reduction=False, data_format=None):
    op_left = conv_cell(hp_dict, type, cell_no, node_no, 'L', inputs, filters, is_reduction, data_format)
    op_right = conv_cell(hp_dict, type, cell_no, node_no, 'R', inputs, filters, is_reduction, data_format)
    name_prefix = f'{type}_C{cell_no}_N{node_no}_'
    return add(name_prefix, [op_left, op_right])


def conv_layer(hp_dict, type, cell_no, inputs, filters, node_num, is_reduction=False, data_format=None):
    name_prefix = f'{type}_C{cell_no}_'

    if inputs[0] == inputs[1]:
        c1 = c2 = CalibrateSize(0, filters, name_prefix, data_format)(inputs[0])
    else:
        c1 = CalibrateSize(0, filters, name_prefix, data_format)(inputs)
        c2 = CalibrateSize(1, filters, name_prefix, data_format)(inputs)
    inputs = [c1, c2]
    all_nodes = [c1, c2]
    for node_no in range(node_num):
        node = conv_node(hp_dict, type, cell_no, node_no, inputs, filters, is_reduction, data_format)
        inputs.append(node)
        all_nodes.append(node)
    cle = ConnectLooseEnd(all_nodes)(all_nodes)

    # df = utils.conv_utils.normalize_data_format(data_format)
    # channel_axis = 1 if df == 'channels_first' else 3
    output = SafeMerge('add', filters, name_prefix, name=name_prefix + 'add_')(cle)
    return output


def stem_op(input, filters, data_format=None):
    conv = Conv2D(
        filters=filters * 3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        data_format=data_format,
        name=f'0_stem_conv2d'
    )
    if input is None:
        input = conv
    else:
        conv(input)
    bn = BatchNormalization(name=f'0_stem_bn')(conv)
    return bn, input


def auxiliary_head(x, data_format=None):
    pass


def classification(x, classes, dropout_rate=0, data_format=None):
    name_prefix = 'classification'
    x = Activation('relu', name=f'{name_prefix}_relu')(x)
    x = GlobalAveragePooling2D(data_format=data_format, name=f'{name_prefix}_global_avgpool')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name=f'{name_prefix}_dropout')(x)
    x = Dense(classes, activation='softmax', name=f'{name_prefix}_logit')(x)
    return x
