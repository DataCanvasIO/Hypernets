# -*- coding:utf-8 -*-
"""

"""

from tensorflow.keras import layers as kl
from hypernets.core.search_space import *
from .utils import compile_layer


class HyperLayer(ModuleSpace):
    def __init__(self, keras_layer_class, space=None, name=None, **hyperparams):
        self.keras_layer_class = keras_layer_class
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _compile(self):
        self.keras_layer = compile_layer(self.space, self.keras_layer_class, self.name, **self.param_values)

    def _forward(self, inputs):
        return self.keras_layer(inputs)

    def _on_params_ready(self):
        pass


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

    def _forward(self, inputs):
        return self.keras_layer


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


class Add(HyperLayer):
    def __init__(self, axis=-1, space=None, name=None, **kwargs):
        if axis != -1:
            kwargs['axis'] = axis
        HyperLayer.__init__(self, kl.Add, space, name, **kwargs)


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


class Conv2D(HyperLayer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, space=None, name=None,
                 **kwargs):
        kwargs['filters'] = filters
        kwargs['kernel_size'] = kernel_size
        if strides is not None:
            kwargs['strides'] = strides
        if padding is not None:
            kwargs['padding'] = padding
        if data_format is not None:
            kwargs['data_format'] = data_format
        HyperLayer.__init__(self, kl.Conv2D, space, name, **kwargs)


class SeparableConv2D(HyperLayer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, space=None, name=None,
                 **kwargs):
        kwargs['filters'] = filters
        kwargs['kernel_size'] = kernel_size
        if strides is not None:
            kwargs['strides'] = strides
        if padding is not None:
            kwargs['padding'] = padding
        if data_format is not None:
            kwargs['data_format'] = data_format
        HyperLayer.__init__(self, kl.SeparableConv2D, space, name, **kwargs)


class AveragePooling2D(HyperLayer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, space=None, name=None,
                 **kwargs):
        if pool_size is not None:
            kwargs['pool_size'] = pool_size
        if strides is not None:
            kwargs['strides'] = strides
        if padding is not None:
            kwargs['padding'] = padding
        if data_format is not None:
            kwargs['data_format'] = data_format
        HyperLayer.__init__(self, kl.AveragePooling2D, space, name, **kwargs)


class MaxPooling2D(HyperLayer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, space=None, name=None,
                 **kwargs):
        if pool_size is not None:
            kwargs['pool_size'] = pool_size
        if strides is not None:
            kwargs['strides'] = strides
        if padding is not None:
            kwargs['padding'] = padding
        if data_format is not None:
            kwargs['data_format'] = data_format
        HyperLayer.__init__(self, kl.MaxPooling2D, space, name, **kwargs)


class GlobalAveragePooling2D(HyperLayer):
    def __init__(self, data_format=None, space=None, name=None, **kwargs):
        if data_format is not None:
            kwargs['data_format'] = data_format
        HyperLayer.__init__(self, kl.GlobalAveragePooling2D, space, name, **kwargs)


class GlobalMaxPooling2D(HyperLayer):
    def __init__(self, data_format=None, space=None, name=None, **kwargs):
        if data_format is not None:
            kwargs['data_format'] = data_format
        HyperLayer.__init__(self, kl.GlobalMaxPooling2D, space, name, **kwargs)


class Cropping2D(HyperLayer):
    def __init__(self, cropping=((0, 0), (0, 0)), data_format=None, space=None, name=None, **kwargs):
        if cropping is not None:
            kwargs['cropping'] = cropping
        if data_format is not None:
            kwargs['data_format'] = data_format
        HyperLayer.__init__(self, kl.Cropping2D, space, name, **kwargs)


class ZeroPadding2D(HyperLayer):
    def __init__(self, padding=(1, 1), data_format=None, space=None, name=None, **kwargs):
        if padding is not None:
            kwargs['padding'] = padding
        if data_format is not None:
            kwargs['data_format'] = data_format
        HyperLayer.__init__(self, kl.ZeroPadding2D, space, name, **kwargs)
