# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from tensorflow.keras import layers, models, utils
from hypernets.core.ops import *
from hypernets.core.search_space import HyperSpace
from hypernets.frameworks.keras.enas_common_ops import sepconv3x3, sepconv5x5, avgpooling3x3, \
    maxpooling3x3, identity
from hypernets.frameworks.keras.layer_weights_cache import LayerWeightsCache
from hypernets.frameworks.keras.layers import Input
from tests import test_output_dir

selection = 0


class LayerChoice(layers.Layer):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self._compiled = False

    def call(self, *inputs):
        # if not self._compiled:
        #     for layer in self.options:
        #         if len(inputs) > 1:
        #             layer.build([inp.shape for inp in inputs])
        #         elif len(inputs) == 1:
        #             layer.build(inputs[0].shape)
        #     self._compiled = True

        global selection
        choice = self.options[selection]
        x = choice(*inputs)
        return x


class SW_Model(models.Model):
    def __init__(self):
        super().__init__()
        self.in_ = layers.Input(shape=(10,))
        options = [layers.Dense(20, activation='relu', name='d1'),
                   layers.Dense(20, activation='relu', name='d2'),
                   layers.Dense(20, activation='relu', name='d3')]
        self.lc = LayerChoice(options)
        self.out_ = layers.Dense(2, activation='softmax')

    def call(self, x):
        # x = self.in_(inputs)
        x = self.lc(x)
        x = self.out_(x)
        return x


class Test_SharingWeights():

    def test_layer_cache(self):
        cache = LayerWeightsCache()

        def get_space(cache):
            space = HyperSpace()
            with space.as_default():
                name_prefix = 'test_'
                filters = 64
                in1 = Input(shape=(28, 28, 1,))
                in2 = Input(shape=(28, 28, 1,))
                ic1 = InputChoice([in1, in2], 1)([in1, in2])
                or1 = ModuleChoice([sepconv5x5(name_prefix, filters),
                                    sepconv3x3(name_prefix, filters),
                                    avgpooling3x3(name_prefix, filters),
                                    maxpooling3x3(name_prefix, filters),
                                    identity(name_prefix)])(ic1)
                space.set_inputs([in1, in2])
                space.weights_cache = cache
                return space

        space = get_space(cache)
        space.assign_by_vectors([1, 0])
        space = space.compile(deepcopy=False)
        assert len(space.weights_cache.cache.items()) == 8
        assert cache.hit_counter == 0
        assert cache.miss_counter == 8

        space = get_space(cache)
        space.assign_by_vectors([1, 1])
        space = space.compile(deepcopy=False)
        assert len(space.weights_cache.cache.items()) == 14
        assert cache.hit_counter == 2
        assert cache.miss_counter == 14

        space = get_space(cache)
        space.assign_by_vectors([1, 0])
        space = space.compile(deepcopy=False)
        assert len(space.weights_cache.cache.items()) == 14
        assert cache.hit_counter == 10
        assert cache.miss_counter == 14

    def test_model(self):
        model = SW_Model()
        utils.plot_model(model, to_file=f'{test_output_dir}/test_ops_0.png', show_shapes=True, expand_nested=True)
        x = np.random.normal(0.0, 1.0, size=(100, 10))
        y = np.random.randint(0, 2, size=(100), dtype='int')

        global selection
        selection = 0

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x, y)
        result = model.evaluate(x, y)
        assert result
        out = model(x)
        assert out.shape == (100, 2)
        # assert model.layers[0].output.shape[1] == 10

        x = np.random.normal(0.0, 1.0, size=(1000, 10))
        y = np.random.randint(0, 2, size=(1000), dtype='int')
        selection = 1
        out = model(x)
        model.fit(x, y)
        result = model.evaluate(x, y)
        assert out.shape == (1000, 2)
