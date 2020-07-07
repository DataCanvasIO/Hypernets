# -*- coding:utf-8 -*-
"""

"""
from tensorflow.keras import layers as kl
from ...core.search_space import ModuleSpace


class EnasHyperLayer(ModuleSpace):
    def __init__(self, compile_fn, filters, name_prefix, space=None, name=None, **hyperparams):
        self.compile_fn = compile_fn
        self.filters = filters
        self.name_prefix = name_prefix
        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _build(self):
        self.is_built = True

    def _compile(self, inputs):
        return self.compile_fn(inputs)

    def _on_params_ready(self):
        pass

    def adjust_layer_sizes(self, x_0, x_1, name_prefix, rep):
        i = 0
        while True:
            x_0_size = x_0.get_shape().as_list()[1]
            x_1_size = x_1.get_shape().as_list()[1]
            if x_0_size == x_1_size:
                break
            elif x_0_size > x_1_size:
                x_0 = self.reduce_output_size(
                    x_0,
                    name_prefix,
                    rep="{0}.{1}".format(str(rep), str(i)),
                    half_current_filters=self.filters // 2)
            elif x_0_size < x_1_size:
                x_1 = self.reduce_output_size(
                    x_1,
                    name_prefix,
                    rep="{0}.{1}".format(str(rep), str(i)),
                    half_current_filters=self.filters // 2)
            i += 1
        return x_0, x_1

    def reduce_output_size(self, inputs, name_prefix, rep,
                           half_current_filters):
        x_0 = kl.AveragePooling2D(
            pool_size=(1, 1),
            strides=(2, 2),
            padding="valid",
            name="{0}_avepool2d_{1}a_".format(name_prefix, rep))(inputs)
        x_0 = kl.Conv2D(
            filters=half_current_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            name="{0}_conv2d_{1}a_".format(name_prefix, rep))(x_0)

        x_1 = kl.ZeroPadding2D(
            padding=((0, 1), (0, 1)),
            name="{0}_zeropad2d_{1}b_".format(name_prefix, rep))(inputs)
        x_1 = kl.Cropping2D(
            cropping=((1, 0), (1, 0)),
            name="{0}_crop2d_{1}b_".format(name_prefix, rep))(x_1)
        x_1 = kl.AveragePooling2D(
            pool_size=(1, 1),
            strides=(2, 2),
            padding="valid",
            name="{0}_avepool2d_{1}b_".format(name_prefix, rep))(x_1)
        x_1 = kl.Conv2D(
            filters=half_current_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            name="{0}_conv2d_{1}b_".format(name_prefix, rep))(x_1)

        x = kl.Concatenate(name="{0}_concat_{1}_".format(name_prefix, rep))(
            [x_0, x_1])
        x = kl.BatchNormalization(name="{0}_bn_{1}_".format(name_prefix, rep))(x)
        return x

    def get_smallest_size_layer(self, x_list):
        """
    get id of a layer with the smallest size in the x_list
    """
        smallest_i = None
        for i in range(len(x_list)):
            if smallest_i is None:
                smallest_i = i
            else:
                if x_list[smallest_i].get_shape().as_list(
                )[1] > x_list[i].get_shape().as_list()[1]:
                    smallest_i = i
        return smallest_i


class Identity(EnasHyperLayer):
    def __init__(self, space=None, name=None, **hyperparams):
        EnasHyperLayer.__init__(self, self._call, None, None, space, name, **hyperparams)

    def _call(self, x):
        return x


class AlignFilters(EnasHyperLayer):
    def __init__(self, filters, name_prefix, space=None, name=None, **hyperparams):
        EnasHyperLayer.__init__(self, self._call, filters, name_prefix, space, name, **hyperparams)

    def _call(self, x):
        if x.get_shape().as_list()[-1] > 1 and x.get_shape().as_list()[-1] != self.filters:
            x = kl.Activation('relu', name=f'{self.name_prefix}relu_')(x)
            x = kl.Conv2D(filters=self.filters,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='same',
                          name=f'{self.name_prefix}conv2d_')(x)
            x = kl.BatchNormalization(name=f'{self.name_prefix}bn_')(x)
            return x
        else:
            return x


class SafeConcatenate(EnasHyperLayer):
    def __init__(self, filters, name_prefix, space=None, name=None, **hyperparams):
        EnasHyperLayer.__init__(self, self._call, filters, name_prefix, space, name, **hyperparams)

    def _call(self, x):
        if isinstance(x, list):
            pv = self.param_values
            if pv.get('name') is None:
                pv['name'] = self.name

            smallest_i = self.get_smallest_size_layer(x)
            resized_x_list = [x[smallest_i]]
            for i in range(len(x)):
                if i == smallest_i:
                    continue
                else:
                    resized_x, _ = self.adjust_layer_sizes(
                        x[i], x[smallest_i], self.name_prefix + 'adjust_size_', i)
                    resized_x_list.append(resized_x)
            return kl.Concatenate(**pv)(resized_x_list)
        else:
            return x


class SafeAdd(EnasHyperLayer):
    def __init__(self, filters, name_prefix, space=None, name=None, **hyperparams):
        EnasHyperLayer.__init__(self, self._call, filters, name_prefix, space, name, **hyperparams)

    def _call(self, x):
        if isinstance(x, list) and len(x) == 2:
            x_0 = x[0]
            x_1 = x[1]
            x_0, x_1 = self.adjust_layer_sizes(x_0, x_1, self.name_prefix + 'adjust_size_', rep=0)
            x = kl.Add(name=f'{self.name_prefix}add_')([x_0, x_1])
            return x
        else:
            return x
