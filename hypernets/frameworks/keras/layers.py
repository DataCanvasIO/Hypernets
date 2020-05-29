# -*- coding:utf-8 -*-
"""

"""

from tensorflow.keras import layers as kl
from hypernets.core.search_space import *


class HyperLayer(ModuleSpace):
    def __init__(self, keras_layer, space=None, name=None, **hyperparams):
        self.keras_layer = keras_layer
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _build(self):
        pv = self.param_values
        if pv.get('name') is None:
            pv['name'] = self.name
        self.compile_fn = self.keras_layer(**pv)
        self.is_built = True

    def _compile(self, inputs):
        return self.compile_fn(inputs)

    def _on_params_ready(self):
        pass
        # self._build()


class Masking(HyperLayer):
    def __init__(self, mask_value=0., space=None, name=None, **kwargs):
        if mask_value != 0.:
            kwargs['mask_value'] = mask_value
        HyperLayer.__init__(self, kl.Masking, space, name, **kwargs)


class Input(HyperLayer):
    def __init__(self, shape, dtype=None, space=None, name=None, **kwargs):
        kwargs['shape'] = shape
        if dtype is not None:
            kwargs['dtype'] = dtype
        HyperLayer.__init__(self, kl.Input, space, name, **kwargs)

    def _compile(self, inputs):
        return self.compile_fn


class Dense(HyperLayer):
    def __init__(self, units, activation=None, use_bias=None, space=None, name=None, **kwargs):
        kwargs['units'] = units
        if activation is not None:
            kwargs['activation'] = activation
        if use_bias is not None:
            kwargs['use_bias'] = use_bias
        HyperLayer.__init__(self, kl.Dense, space, name, **kwargs)


class Embedding(HyperLayer):
    def __init__(self, input_dim, output_dim, space=None, name=None, **kwargs):
        kwargs['input_dim'] = input_dim
        kwargs['output_dim'] = output_dim
        HyperLayer.__init__(self, kl.Embedding, space, name, **kwargs)


class Dropout(HyperLayer):
    def __init__(self, rate, space=None, name=None, **kwargs):
        kwargs['rate'] = rate
        HyperLayer.__init__(self, kl.Dropout, space, name, **kwargs)


class SpatialDropout1D(HyperLayer):
    def __init__(self, rate, space=None, name=None, **kwargs):
        kwargs['rate'] = rate
        HyperLayer.__init__(self, kl.SpatialDropout1D, space, name, **kwargs)


class SpatialDropout2D(HyperLayer):
    def __init__(self, rate, data_format=None, space=None, name=None, **kwargs):
        kwargs['rate'] = rate
        if data_format is not None:
            kwargs['data_format'] = data_format
        HyperLayer.__init__(self, kl.SpatialDropout2D, space, name, **kwargs)


class SpatialDropout3D(HyperLayer):
    def __init__(self, rate, data_format=None, space=None, name=None, **kwargs):
        kwargs['rate'] = rate
        if data_format is not None:
            kwargs['data_format'] = data_format
        HyperLayer.__init__(self, kl.SpatialDropout3D, space, name, **kwargs)


class BatchNormalization(HyperLayer):
    def __init__(self, space=None, name=None, **kwargs):
        HyperLayer.__init__(self, kl.BatchNormalization, space, name, **kwargs)


class Concatenate(HyperLayer):
    def __init__(self, axis=-1, space=None, name=None, **kwargs):
        if axis != -1:
            kwargs['axis'] = axis
        HyperLayer.__init__(self, kl.Concatenate, space, name, **kwargs)


class Reshape(HyperLayer):
    def __init__(self, target_shape, space=None, name=None, **kwargs):
        kwargs['target_shape'] = target_shape
        HyperLayer.__init__(self, kl.Reshape, space, name, **kwargs)


class Flatten(HyperLayer):
    def __init__(self, data_format=None, space=None, name=None, **kwargs):
        if data_format is not None:
            kwargs['data_format'] = data_format
        HyperLayer.__init__(self, kl.Flatten, space, name, **kwargs)


class Activation(HyperLayer):
    def __init__(self, activation, space=None, name=None, **kwargs):
        kwargs['activation'] = activation
        HyperLayer.__init__(self, kl.Activation, space, name, **kwargs)
