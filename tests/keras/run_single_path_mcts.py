# -*- coding:utf-8 -*-
"""

"""
import tensorflow as tf

from hypernets.core.callbacks import SummaryCallback
from hypernets.core.ops import *
from hypernets.frameworks.keras.enas_micro import enas_micro_search_space
from hypernets.frameworks.keras.one_shot_model import OneShotModel
from hypernets.searchers.mcts_searcher import MCTSSearcher
from hypernets.searchers.random_searcher import RandomSearcher

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))
# sample for speed up
samples = 100

searcher = MCTSSearcher(lambda: enas_micro_search_space(arch='NNRNNR', hp_dict={}), optimize_direction='max')
one_shot_sampler = RandomSearcher(lambda: enas_micro_search_space(arch='NNRNNR', hp_dict={}), optimize_direction='max')

model = OneShotModel(searcher,
                     optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'],
                     epochs=3,
                     batch_size=64,
                     controller_train_per_epoch=False,  # Single path
                     callbacks=[SummaryCallback()],
                     one_shot_train_sampler=one_shot_sampler,  # uniform sampler
                     visualization=False)

model.search(x_train[:samples], y_train[:samples], x_test[:int(samples / 10)], y_test[:int(samples / 10)],
             max_trails=100, epochs=1, callbacks=[])
assert model.best_model
