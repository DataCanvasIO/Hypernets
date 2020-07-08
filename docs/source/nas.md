# Neural Architecture Search

Deep Learning has enabled remarkable progress over the last years on a variety of tasks, such as CV, NLP, and machine translation.  It is crucial to discover novel neural architectures, but currently it have mostly been developed manually by human experts.  Neural Architecture Search (NAS) has emerged as a promising tool to alleviate human effort in this trial and error design process.

NAS has demonstrated much success in automating neural network architecture design for various tasks, such as image recognition and language modeling. Representative works include NASNet, ENAS, DARTS, ProxylessNAS, One-Shot NAS, Regularized Evolution, AlphaX, etc.

However, most of these works are usually for specific use-cases, and their search space, search strategy and estimation strategy are often intertwined, making it difficult to reuse the code and make further innovations on it.

In Hypernets, we propose an abstract architecture, fully decouple Search Space, Search Strategy, and Performance Estimation Strategy so that each part is relatively independent and can be reused to accelerate innovations and engineering of NAS algorithms.


The 3 problems of NAS: Search Space, Search Strategy, and Performance Estimation Strategy,  which correspond to `HyperSpace`, `Searcher`, and `Estimator` in Hypernets respectively.

<p align="center">
<img src="https://raw.githubusercontent.com/DataCanvasIO/Hypernets/master/docs/source/images/abstract_illustration_of_nas.png" width="100%"/>
</p>

## Define A DNN Search Space

```python
# define a DNN search space

from hypernets.frameworks.keras.layers import Dense, Input, BatchNormalization, Dropout, Activation
from hypernets.core.search_space import HyperSpace, Bool, Choice, Real, Dynamic
from hypernets.core.ops import Permutation, Sequential, Optional, Repeat
import itertools

def dnn_block(hp_dnn_units, hp_reduce_factor, hp_seq, hp_use_bn, hp_dropout, hp_activation, step):
    
    # The value of a `Dynamic` is computed as a function of the values of the ParameterSpace it depends on.
    block_units = Dynamic(
        lambda_fn=lambda units, reduce_factor: units if step == 0 else units * (reduce_factor ** step),
        units=hp_dnn_units, reduce_factor=hp_reduce_factor)

    dense = Dense(units=block_units)
    act = Activation(activation=hp_activation)
    optional_bn = Optional(BatchNormalization(), keep_link=True, hp_opt=hp_use_bn)
    dropout = Dropout(rate=hp_dropout)

    # Use `Permutation` to try different arrangements of act, optional_bn, dropout
    # optional_bn is optional module and will be skipped when hp_use_bn is False
    perm_act_bn_dropout = Permutation([act, optional_bn, dropout], hp_seq=hp_seq)
    
    # Use `Sequential` to connect dense and perm_act_bn_dropout in order
    seq = Sequential([dense, perm_act_bn_dropout])
    return seq


def dnn_search_space(input_shape, output_units, output_activation, units_choices=[200, 500, 1000],
                     reduce_facotr_choices=[1, 0.8, 0.5], layer_num_choices=[2, 3, 4],
                     activation_choices=['relu', 'tanh']):
    space = HyperSpace()
    with space.as_default():
        hp_dnn_units = Choice(units_choices)
        hp_reduce_factor = Choice(reduce_facotr_choices)
        hp_use_bn = Bool()
        if len(activation_choices) == 1:
            hp_activation = activation_choices[0]
        else:
            hp_activation = Choice(activation_choices)

        hp_dropout = Real(0., 0.5, step=0.1)
        hp_seq = Choice([seq for seq in itertools.permutations(range(3))])

        input = Input(shape=input_shape)
        backbone = Repeat(
            lambda step: dnn_block(hp_dnn_units, hp_reduce_factor, hp_seq, hp_use_bn, hp_dropout, hp_activation, step),
            repeat_num_choices=layer_num_choices)(input)
        output = Dense(units=output_units, activation=output_activation)(backbone)
    return space


# Search the best model in search space defined above 
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.callbacks import SummaryCallback
from hypernets.frameworks.keras.models import HyperKeras
import numpy as np

rs = RandomSearcher(lambda: dnn_search_space(input_shape=10, output_units=2, output_activation='sigmoid'),
                    optimize_direction='max')
hk = HyperKeras(rs, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                callbacks=[SummaryCallback()])

x = np.random.randint(0, 10000, size=(100, 10))
y = np.random.randint(0, 2, size=(100), dtype='int')

hk.search(x, y, x, y, max_trails=3)
assert hk.best_model
```

## Define A CNN Search Space
```python
from hypernets.frameworks.keras.layers import Dense, Input, BatchNormalization, Activation, \
    Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from hypernets.core.search_space import HyperSpace, Bool, Choice, Dynamic
from hypernets.core.ops import Permutation, Sequential, Optional, Repeat, Or
import itertools


def conv_block(block_no, hp_pooling, hp_filters, hp_kernel_size, hp_bn_act, hp_use_bn, hp_activation, strides=(1, 1)):
    def conv_bn(step):
        conv = Conv2D(filters=conv_filters, kernel_size=hp_kernel_size, strides=strides, padding='same')
        act = Activation(activation=hp_activation)
        optional_bn = Optional(BatchNormalization(), keep_link=True, hp_opt=hp_use_bn)

        # Use `Permutation` to try different arrangements of act, optional_bn
        # optional_bn is optional module and will be skipped when hp_use_bn is False
        perm_act_bn = Permutation([optional_bn, act], hp_seq=hp_bn_act)
        seq = Sequential([conv, perm_act_bn])
        return seq

    if block_no < 2:
        repeat_num_choices = [2]
        multiplier = 1
    else:
        repeat_num_choices = [3, 4, 5]
        multiplier = 2 ** (block_no - 1)

    conv_filters = Dynamic(lambda filters: filters * multiplier, filters=hp_filters)
    conv = Repeat(conv_bn, repeat_num_choices=repeat_num_choices)
    pooling = Or([MaxPooling2D(padding='same'), AveragePooling2D(padding='same')], hp_or=hp_pooling)
    block = Sequential([conv, pooling])
    return block


def cnn_search_space(input_shape, output_units, output_activation='softmax', block_num_choices=[2, 3, 4, 5, 6],
                     activation_choices=['relu'], filters_choices=[32, 64], kernel_size_choices=[(1, 1), (3, 3)]):
    space = HyperSpace()
    with space.as_default():
        hp_use_bn = Bool()
        hp_pooling = Choice(list(range(2)))
        hp_filters = Choice(filters_choices)
        hp_kernel_size = Choice(kernel_size_choices)
        hp_fc_units = Choice([1024, 2048, 4096])
        if len(activation_choices) == 1:
            hp_activation = activation_choices[0]
        else:
            hp_activation = Choice(activation_choices)
        hp_bn_act = Choice([seq for seq in itertools.permutations(range(2))])

        input = Input(shape=input_shape)
        blocks = Repeat(
            lambda step: conv_block(
                block_no=step,
                hp_pooling=hp_pooling,
                hp_filters=hp_filters,
                hp_kernel_size=hp_kernel_size,
                hp_use_bn=hp_use_bn,
                hp_activation=hp_activation,
                hp_bn_act=hp_bn_act),
            repeat_num_choices=block_num_choices)(input)
        x = Flatten()(blocks)
        x = Dense(units=hp_fc_units, activation=hp_activation, name='fc1')(x)
        x = Dense(units=hp_fc_units, activation=hp_activation, name='fc2')(x)
        x = Dense(output_units, activation=output_activation, name='predictions')(x)
    return space


# Search the best model in search space defined above on mnist dataset

from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.callbacks import SummaryCallback
from hypernets.frameworks.keras.models import HyperKeras
import numpy as np
import tensorflow as tf

rs = RandomSearcher(
    lambda: cnn_search_space(input_shape=(28, 28, 1),
                             output_units=10,
                             output_activation='softmax',
                             block_num_choices=[2, 3, 4, 5],
                             filters_choices=[32, 64, 128],
                             kernel_size_choices=[(1, 1), (3, 3)]),
    optimize_direction='max')
hk = HyperKeras(rs, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
                callbacks=[SummaryCallback()])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

hk.search(x_train, y_train, x_test, y_test, max_trails=10, epochs=10)
assert hk.best_model
```

## Define An ENAS Micro Search Space

```python
# define an ENAS micro search space

from hypernets.frameworks.keras.enas_layers import SafeConcatenate, Identity, CalibrateSize
from hypernets.frameworks.keras.layers import BatchNormalization, Activation, Add, MaxPooling2D, AveragePooling2D, \
    SeparableConv2D
from hypernets.core.ops import Or, InputChoice, ConnectLooseEnd
from hypernets.core.search_space import ModuleSpace


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


def add(x1, x2, name_prefix, filters):
    return Add(name=f'{name_prefix}add_')([x1, x2])


def conv_cell(hp_dict, type, cell_no, node_no, left_or_right, inputs, filters, is_reduction=False, data_format=None):
    assert isinstance(inputs, list)
    assert all([isinstance(m, ModuleSpace) for m in inputs])
    name_prefix = f'{type}_C{cell_no}_N{node_no}_{left_or_right}_'

    input_choice_key = f'{type[2:]}_N{node_no}_{left_or_right}_input_choice'
    op_choice_key = f'{type[2:]}_N{node_no}_{left_or_right}_op_choice'
    hp_choice = hp_dict.get(input_choice_key)
    ic1 = InputChoice(inputs, 1, hp_choice=hp_choice)(inputs)
    if hp_choice is None:
        hp_dict[input_choice_key] = ic1.hp_choice

    # hp_strides = Dynamic(lambda_fn=lambda choice: (2, 2) if is_reduction and choice[0] <= 1 else (1, 1),
    #                      choice=ic1.hp_choice)
    hp_strides = (1, 1)
    hp_or = hp_dict.get(op_choice_key)
    or1 = Or([sepconv5x5(name_prefix, filters, strides=hp_strides, data_format=data_format),
              sepconv3x3(name_prefix, filters, strides=hp_strides, data_format=data_format),
              avgpooling3x3(name_prefix, filters, strides=hp_strides, data_format=data_format),
              maxpooling3x3(name_prefix, filters, strides=hp_strides, data_format=data_format),
              identity(name_prefix)], hp_or=hp_or)(ic1)

    if hp_or is None:
        hp_dict[op_choice_key] = or1.hp_or

    return or1


def conv_node(hp_dict, type, cell_no, node_no, inputs, filters, is_reduction=False, data_format=None):
    op_left = conv_cell(hp_dict, type, cell_no, node_no, 'L', inputs, filters, is_reduction, data_format)
    op_right = conv_cell(hp_dict, type, cell_no, node_no, 'R', inputs, filters, is_reduction, data_format)
    name_prefix = f'{type}_C{cell_no}_N{node_no}_'
    return add(op_left, op_right, name_prefix, filters)


def conv_layer(hp_dict, type, cell_no, inputs, filters, node_num, is_reduction=False, data_format=None):
    name_prefix = f'{type}_C{cell_no}_'

    if inputs[0] == inputs[1]:
        c1 = c2 = CalibrateSize(0, filters, name_prefix, data_format)(inputs[0])
    else:
        c1 = CalibrateSize(0, filters, name_prefix, data_format)(inputs)
        c2 = CalibrateSize(1, filters, name_prefix, data_format)(inputs)
    inputs = [c1, c2]
    all_nodes = []
    for node_no in range(node_num):
        node = conv_node(hp_dict, type, cell_no, node_no, inputs, filters, is_reduction, data_format)
        inputs.append(node)
        all_nodes.append(node)
    cle = ConnectLooseEnd(all_nodes)(all_nodes)
    concat = SafeConcatenate(filters, name_prefix, name=name_prefix + 'concat_')(cle)
    return concat


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



# Search the best model in search space defined above on mnist dataset
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.callbacks import SummaryCallback
from hypernets.frameworks.keras.models import HyperKeras
import numpy as np
import tensorflow as tf

rs = RandomSearcher(lambda: enas_micro_search_space(arch='NRNR', input_shape=(28, 28, 1), init_filters=64, node_num=4),optimize_direction='max')
hk = HyperKeras(rs, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
                callbacks=[SummaryCallback()])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

hk.search(x_train, y_train, x_test, y_test, max_trails=10, epochs=10)
assert hk.best_model
```

## API Reference

```python
# api

```


[1] Elsken T, Metzen J H, Hutter F. Neural architecture search: A survey[J]. arXiv preprint arXiv:1808.05377, 2018.
