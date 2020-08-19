# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.core.search_space import HyperSpace
from tensorflow.keras.models import Model

def keras_model(self, deepcopy=True):
    compiled_space, outputs = self.compile_and_forward(deepcopy=deepcopy)
    inputs = compiled_space.get_inputs()
    model = Model(inputs=[input.output for input in inputs],
                  outputs=outputs)
    return model


HyperSpace.keras_model = keras_model