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
        reduction = 1
    else:
        repeat_num_choices = [3, 4, 5]
        reduction = 2 ** (block_no - 1)

    conv_filters = Dynamic(lambda filters: filters * reduction, filters=hp_filters)
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

```

## API Reference

```python
# api

```


[1] Elsken T, Metzen J H, Hutter F. Neural architecture search: A survey[J]. arXiv preprint arXiv:1808.05377, 2018.
