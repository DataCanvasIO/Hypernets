# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.core.callbacks import SummaryCallback
from hypernets.core.ops import *
from hypernets.frameworks.keras.enas_micro import enas_micro_search_space
from hypernets.frameworks.keras.hyper_keras import HyperKeras
from hypernets.frameworks.keras.layer_weights_cache import LayerWeightsCache
from hypernets.searchers.random_searcher import RandomSearcher

import tensorflow as tf
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))
# sample for speed up
samples = 10


class Test_KerasLayer():
    def test_conv2d(self):
        x = layers.Conv2D(32, (1, 1))(x_train[:samples])
        assert x.shape == (10, 28, 28, 32)
        x = layers.Conv2D(32, (2, 2), padding='same')(x_train[:samples])
        assert x.shape == (10, 28, 28, 32)
        x = layers.Conv2D(32, (2, 2), strides=(1, 1), padding='same')(x_train[:samples])
        assert x.shape == (10, 28, 28, 32)
        x = layers.Conv2D(32, (2, 2), strides=(2, 2), padding='same')(x_train[:samples])
        assert x.shape == (10, 14, 14, 32)
        x = layers.Conv2D(32, (2, 2), strides=(2, 1), padding='same')(x_train[:samples])
        assert x.shape == (10, 14, 28, 32)

    def test_pooling(self):
        conv_x = layers.Conv2D(32, (1, 1))(x_train[:samples])
        assert conv_x.shape == (10, 28, 28, 32)
        x = layers.AveragePooling2D(pool_size=3, strides=1, padding='same')(conv_x)
        assert x.shape == (10, 28, 28, 32)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv_x)
        assert x.shape == (10, 14, 14, 32)
        x = layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(conv_x)
        assert x.shape == (10, 28, 28, 32)
