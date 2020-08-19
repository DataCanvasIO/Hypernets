# -*- coding:utf-8 -*-
"""

"""
from tensorflow.keras import layers as kl, utils
from ...core.search_space import ModuleSpace
from .utils import compile_layer


class FilterAlignment(kl.Layer):
    def __init__(self, name_prefix, filters, data_format, **kwargs):
        self.name_prefix = name_prefix
        self.filters = filters
        self.data_format = data_format
        self.relu = kl.ReLU(name=f'{name_prefix}_filters_aligning_relu_')
        self.conv2d = kl.Conv2D(filters, (1, 1), padding='same', name=f'{name_prefix}_filters_aligning_conv2d_',
                                data_format=data_format)
        self.bn = kl.BatchNormalization(name=f'{name_prefix}_filters_aligning_bn_')
        super(FilterAlignment, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = self.relu(inputs)
        x = self.conv2d(x)
        x = self.bn(x)
        return x

    def get_config(self, ):
        config = {'name_prefix': self.name_prefix, 'filters': self.filters, 'data_format': self.data_format}
        base_config = super(FilterAlignment, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FactorizedReduction_K(kl.Layer):
    def __init__(self, name_prefix, filters, data_format, **kwargs):
        self.name_prefix = name_prefix
        self.filters = filters
        self.data_format = data_format
        self.relu = kl.ReLU(name=f'{name_prefix}relu_')
        self.x0_avgpool = kl.AveragePooling2D(
            pool_size=(1, 1),
            strides=(2, 2),
            padding="valid",
            data_format=self.data_format,
            name=f'{name_prefix}avepool2d_a_')
        self.x0_conv = kl.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            data_format=self.data_format,
            name=f"{name_prefix}conv2d_a_")
        self.x1_zeropad = kl.ZeroPadding2D(
            padding=((0, 1), (0, 1)),
            data_format=self.data_format,
            name=f"{name_prefix}zeropad2d_b_")
        self.x1_cropping = kl.Cropping2D(
            cropping=((1, 0), (1, 0)),
            data_format=self.data_format,
            name=f"{name_prefix}crop2d_b_")
        self.x1_avgpool = kl.AveragePooling2D(
            pool_size=(1, 1),
            strides=(2, 2),
            padding="valid",
            data_format=self.data_format,
            name=f"{name_prefix}avepool2d_b_")
        self.x1_conv = kl.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            data_format=self.data_format,
            name=f"{name_prefix}conv2d_b_")
        self.concat = kl.Concatenate(name=f"{name_prefix}concat_")
        self.bn = kl.BatchNormalization(name=f"{name_prefix}bn_")
        super(FactorizedReduction_K, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = self.relu(inputs)
        x_0 = self.x0_avgpool(x)
        x_0 = self.x0_conv(x_0)
        x_1 = self.x1_zeropad(x)
        x_1 = self.x1_cropping(x_1)
        x_1 = self.x1_avgpool(x_1)
        x_1 = self.x0_conv(x_1)
        x = self.concat([x_0, x_1])
        x = self.bn(x)
        return x

    def get_config(self, ):
        config = {'name_prefix': self.name_prefix, 'filters': self.filters, 'data_format': self.data_format}
        base_config = super(FactorizedReduction_K, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EnasHyperLayer(ModuleSpace):
    def __init__(self, filters, name_prefix, data_format=None, space=None, name=None, **hyperparams):
        self.filters = filters
        self.name_prefix = name_prefix
        self.data_format = data_format
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _compile(self):
        pass

    def _on_params_ready(self):
        pass


class Identity(EnasHyperLayer):
    def __init__(self, space=None, name=None, **hyperparams):
        EnasHyperLayer.__init__(self, None, None, None, space, name, **hyperparams)

    def _forward(self, inputs):
        return inputs


class FactorizedReduction(EnasHyperLayer):
    def __init__(self, filters, name_prefix, data_format=None, space=None, name=None, **hyperparams):
        EnasHyperLayer.__init__(self, filters, name_prefix, data_format, space, name,
                                **hyperparams)

    def _compile(self):
        self.factorized_reduction = compile_layer(search_space=self.space,
                                                  layer_class=FactorizedReduction_K,
                                                  name=f'{self.name_prefix}reduction_',
                                                  name_prefix=f'{self.name_prefix}reduction_',
                                                  filters=self.filters // 2,
                                                  data_format=self.data_format)

    def _forward(self, inputs):
        return self.factorized_reduction(inputs)


class CalibrateSize(EnasHyperLayer):
    def __init__(self, no, filters, name_prefix, data_format=None, space=None, name=None, **hyperparams):
        self.no = no
        EnasHyperLayer.__init__(self, filters, name_prefix, data_format, space, name,
                                **hyperparams)

    def _compile(self):
        self.filter_alignment_input0 = compile_layer(search_space=self.space,
                                                     layer_class=FilterAlignment,
                                                     name=f'{self.name_prefix}input0_filter_alignment',
                                                     name_prefix=self.name_prefix + 'input0',
                                                     filters=self.filters,
                                                     data_format=self.data_format)
        self.filter_alignment_input1 = compile_layer(search_space=self.space,
                                                     layer_class=FilterAlignment,
                                                     name=f'{self.name_prefix}input1_filter_alignment',
                                                     name_prefix=self.name_prefix + 'input1',
                                                     filters=self.filters,
                                                     data_format=self.data_format)
        self.factorized_reduction = compile_layer(search_space=self.space,
                                                  layer_class=FactorizedReduction_K,
                                                  name=self.name_prefix + 'input0_factorized_reduction_',
                                                  name_prefix=self.name_prefix + 'input0_factorized_reduction_',
                                                  filters=self.filters // 2,
                                                  data_format=self.data_format)

    def get_height_or_width(self, x):
        return x.get_shape().as_list()[2]

    def get_channels(self, x):
        if self.data_format in {None, 'channels_last'}:
            return x.get_shape().as_list()[3]
        else:
            return x.get_shape().as_list()[1]

    def _forward(self, inputs):
        if isinstance(inputs, list):
            assert len(inputs) == 2
            hw = [self.get_height_or_width(l) for l in inputs]
            c = [self.get_channels(l) for l in inputs]
            if self.no == 0:
                x = inputs[0]
                if hw[0] != hw[1]:
                    assert hw[0] == 2 * hw[1]
                    x = self.factorized_reduction(x)
                elif c[0] != self.filters:
                    x = self.filter_alignment_input0(x)
            else:
                x = inputs[1]
                if c[1] != self.filters:
                    x = self.filter_alignment_input1(x)
            return x
        else:
            x = inputs
            c = self.get_channels(x)
            if c != self.filters:
                x = self.filter_alignment_input0(x)
            return x


class SafeMerge(EnasHyperLayer):
    def __init__(self, operation, filters, name_prefix, data_format=None, space=None, name=None, **hyperparams):
        self.operation = operation.lower()
        EnasHyperLayer.__init__(self, filters, name_prefix, data_format, space, name, **hyperparams)

    def _forward(self, inputs):
        if isinstance(inputs, list):
            pv = self.param_values
            if pv.get('name') is None:
                pv['name'] = self.name
            if self.operation == 'add':
                return kl.Add(name=pv['name'])(inputs)
            elif self.operation == 'concat':
                return kl.Concatenate(**pv)(inputs)
            else:
                raise ValueError(f'Not supported operation:{self.operation}')
        else:
            return inputs
