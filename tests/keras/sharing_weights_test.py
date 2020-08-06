# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from tensorflow.keras import layers, models, utils
import tensorflow as tf
import numpy as np

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

    # def test_call(self):
    #     global selection
    #     selection = 0
    #     options = [layers.Dense(20, activation='relu', name='d1'),
    #                layers.Dense(20, activation='relu', name='d2'),
    #                layers.Dense(20, activation='relu', name='d3')]
    #     x = np.random.normal(0.0, 1.0, size=(100, 10))
    #     lc = LayerChoice(options)(x)
    #     d1 = layers.Dense(20, activation='relu', name='d1')
    #     out_ = layers.Dense(2, activation='softmax')(lc)
    #     model = models.Model(inputs=lc.i, outputs=out_)
    #     assert model

    def test_model(self):
        model = SW_Model()
        utils.plot_model(model, to_file='test_ops_0.png', show_shapes=True, expand_nested=True)
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
        # assert model.layers[1].output.shape[1] == 20
