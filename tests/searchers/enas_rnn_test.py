# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from hypernets.core.search_space import HyperSpace
from hypernets.frameworks.keras.enas_common_ops import conv_layer
from hypernets.frameworks.keras.layers import Input
from hypernets.searchers.enas_rl_searcher import RnnController

baseline_decay = 0.999


class Test_EnasRnnController():
    def get_space(self):
        hp_dict = {}
        space = HyperSpace()
        with space.as_default():
            filters = 64
            in1 = Input(shape=(28, 28, 1,))
            conv_layer(hp_dict, 'normal', 0, [in1, in1], filters, 5)
            space.set_inputs(in1)
            return space

    def test_call(self):
        rc = RnnController(search_space_fn=self.get_space)
        rc.reset()
        out1 = rc.sample()
        out2 = rc.sample()
        assert out1
        assert out2

    def test_train(self):
        self.reward = 0.6
        self.baseline = 0.
        self.rc = RnnController(search_space_fn=self.get_space)
        self.optim = Adam()

        grads_list = []
        for step in range(10):
            with tf.GradientTape() as tape:
                self.rc.reset()
                out1 = self.rc.sample()
                self.reward += 0.01
                self.baseline = self.baseline * baseline_decay + self.reward * (1 - baseline_decay)
                loss = self.rc.sample_log_prob * (self.reward - self.baseline)
                # loss += skip_weight * self.mutator.sample_skip_penalty
            grads = tape.gradient(loss, self.rc.trainable_variables)
            self.optim.apply_gradients(zip(grads, self.rc.trainable_variables))

        assert self.rc
