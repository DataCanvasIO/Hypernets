# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.core.search_space import HyperSpace
from tensorflow.keras.models import Model


def keras_model(self):
    compiled_space = self.compile_space()
    inputs = compiled_space.get_inputs()
    outputs = compiled_space.get_outputs()
    model = Model(inputs=[input.output for input in inputs],
                  outputs=[output.output for output in outputs])
    return model


HyperSpace.keras_model = keras_model
