# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.frameworks.keras.layers import *
from hypernets.frameworks.keras.hyper_keras import HyperKeras
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection

def get_space():
    space = HyperSpace()
    with space.as_default():
        in1 = Input(shape=(10,))
        dense1 = Dense(10, activation=Choice(['relu', 'tanh', None]), use_bias=Bool())(in1)
        bn1 = BatchNormalization()(dense1)
        dropout1 = Dropout(Choice([0.3, 0.4, 0.5]))(bn1)
        output = Dense(2, activation='softmax', use_bias=True)(dropout1)
    return space


rs = RandomSearcher(get_space, optimize_direction=OptimizeDirection.Maximize)
hk = HyperKeras(rs, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                callbacks=[SummaryCallback()])

x = np.random.randint(0, 10000, size=(100, 10))
y = np.random.randint(0, 2, size=(100), dtype='int')

hk.search(x, y, x, y, max_trails=3)
assert hk.best_model
best_trial = hk.get_best_trail()

estimator = hk.final_train(best_trial.space_sample, x, y)
score = estimator.predict(x)
result = estimator.evaluate(x, y)
assert len(score) == 100
assert result