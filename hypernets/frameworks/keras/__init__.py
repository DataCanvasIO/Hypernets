# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.core.search_space import HyperSpace
from tensorflow.keras.models import Model


def keras_model(self):
    if not self._is_compiled:
        self.compile_space()
    inputs = self.get_inputs()
    outputs = self.get_outputs()
    model = Model(inputs=[input.output for input in inputs],
                  outputs=[output.output for output in outputs])
    return model


HyperSpace.keras_model = keras_model
