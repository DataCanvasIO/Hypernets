# -*- coding:utf-8 -*-
"""

"""
import tensorflow as tf

from hypernets.core.callbacks import SummaryCallback
from hypernets.core.ops import *
from hypernets.frameworks.keras.enas_micro import enas_micro_search_space
from hypernets.frameworks.keras.hyper_keras import HyperKeras
from hypernets.searchers.random_searcher import RandomSearcher

rs = RandomSearcher(
    lambda: enas_micro_search_space(arch='NNRNNR', hp_dict={}),
    optimize_direction='max')
hk = HyperKeras(rs, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
                callbacks=[SummaryCallback()])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))

# sample for speed up
samples = 100
hk.search(x_train[:samples], y_train[:samples], x_test[:int(samples / 10)], y_test[:int(samples / 10)],
          max_trails=10, epochs=1)
assert hk.best_model
