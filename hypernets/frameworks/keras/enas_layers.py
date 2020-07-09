# -*- coding:utf-8 -*-
"""

"""
from tensorflow.keras import layers as kl
from ...core.search_space import ModuleSpace


class EnasHyperLayer(ModuleSpace):
    def __init__(self, compile_fn, filters, name_prefix, data_format=None, space=None, name=None, **hyperparams):
        self.compile_fn = compile_fn
        self.filters = filters
        self.name_prefix = name_prefix
        self.data_format = data_format
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _build(self):
        self.is_built = True

    def _compile(self, inputs):
        return self.compile_fn(inputs)

    def _on_params_ready(self):
        pass

    def get_height_or_width(self, x):
        return x.get_shape().as_list()[2]

    def get_channels(self, x):
        if self.data_format in {None, 'channels_last'}:
            return x.get_shape().as_list()[3]
        else:
            return x.get_shape().as_list()[1]

    def filters_aligning(self, x, input_no):
        x = kl.ReLU(name=f'{self.name_prefix}{input_no}_filters_aligning_relu_')(x)
        x = kl.Conv2D(self.filters, (1, 1), padding='same',
                      name=f'{self.name_prefix}{input_no}_filters_aligning_conv2d_',
                      data_format=self.data_format)(x)
        x = kl.BatchNormalization(name=f'{self.name_prefix}{input_no}_filters_aligning_bn_')(x)
        return x

    def factorized_reduction(self, x, name_prefix):
        half_filters = self.filters // 2
        x = kl.ReLU(name=f'{self.name_prefix}relu_')(x)
        x_0 = kl.AveragePooling2D(
            pool_size=(1, 1),
            strides=(2, 2),
            padding="valid",
            data_format=self.data_format,
            name=f'{name_prefix}avepool2d_a_')(x)
        x_0 = kl.Conv2D(
            filters=half_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            data_format=self.data_format,
            name=f"{name_prefix}conv2d_a_")(x_0)

        x_1 = kl.ZeroPadding2D(
            padding=((0, 1), (0, 1)),
            data_format=self.data_format,
            name=f"{name_prefix}zeropad2d_b_")(x)
        x_1 = kl.Cropping2D(
            cropping=((1, 0), (1, 0)),
            data_format=self.data_format,
            name=f"{name_prefix}crop2d_b_")(x_1)
        x_1 = kl.AveragePooling2D(
            pool_size=(1, 1),
            strides=(2, 2),
            padding="valid",
            data_format=self.data_format,
            name=f"{name_prefix}avepool2d_b_")(x_1)
        x_1 = kl.Conv2D(
            filters=half_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            data_format=self.data_format,
            name=f"{name_prefix}conv2d_b_")(x_1)

        x = kl.Concatenate(name=f"{name_prefix}concat_")([x_0, x_1])
        x = kl.BatchNormalization(name=f"{name_prefix}bn_")(x)
        return x


class Identity(EnasHyperLayer):
    def __init__(self, space=None, name=None, **hyperparams):
        EnasHyperLayer.__init__(self, self._call, None, None, None, space, name, **hyperparams)

    def _call(self, x):
        return x


class FactorizedReduction(EnasHyperLayer):
    def __init__(self, filters, name_prefix, data_format=None, space=None, name=None, **hyperparams):
        EnasHyperLayer.__init__(self, self._call, filters, name_prefix, data_format, space, name,
                                **hyperparams)

    def _call(self, x):
        return self.factorized_reduction(x, f'{self.name_prefix}reduction_')


class CalibrateSize(EnasHyperLayer):
    def __init__(self, no, filters, name_prefix, data_format=None, space=None, name=None, **hyperparams):
        self.no = no
        EnasHyperLayer.__init__(self, self.calibrate_size, filters, name_prefix, data_format, space, name,
                                **hyperparams)

    def calibrate_size(self, inputs):
        if isinstance(inputs, list):
            assert len(inputs) == 2
            hw = [self.get_height_or_width(l) for l in inputs]
            c = [self.get_channels(l) for l in inputs]
            if self.no == 0:
                x = inputs[0]
                if hw[0] != hw[1]:
                    assert hw[0] == 2 * hw[1]
                    x = self.factorized_reduction(x, self.name_prefix + 'input0_factorized_reduction_')
                elif c[0] != self.filters:
                    x = self.filters_aligning(x, 'input0')
            else:
                x = inputs[1]
                if c[1] != self.filters:
                    x = self.filters_aligning(x, 'input1')
            return x
        else:
            x = inputs
            c = self.get_channels(x)
            if c != self.filters:
                x = self.filters_aligning(x, 'input0')
            return x


class SafeConcatenate(EnasHyperLayer):
    def __init__(self, filters, name_prefix, data_format=None, space=None, name=None, **hyperparams):
        EnasHyperLayer.__init__(self, self._call, filters, name_prefix, data_format, space, name, **hyperparams)

    def _call(self, x):
        if isinstance(x, list):
            pv = self.param_values
            if pv.get('name') is None:
                pv['name'] = self.name
            return kl.Concatenate(**pv)(x)
        else:
            return x
