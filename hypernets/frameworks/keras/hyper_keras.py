# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.model.hyper_model import HyperModel
from hypernets.model.estimator import Estimator
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers, utils
import numpy as np
import gc
from .layer_weights_cache import LayerWeightsCache


def keras_model(self):
    compiled_space, _ = self.compile_and_forward()
    inputs = compiled_space.get_inputs()
    outputs = compiled_space.get_outputs()
    model = models.Model(inputs=[input.output for input in inputs],
                         outputs=[output.output for output in outputs])
    return model


class ss:
    def __init__(self):
        self.conv = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name=f'0_stem_conv2d'
        )
        self.bn = layers.BatchNormalization(name=f'0_stem_bn')
        name_prefix = 'output'

        self.act = layers.Activation('relu', name=f'{name_prefix}_relu')
        self.gap = layers.GlobalAveragePooling2D(name=f'{name_prefix}_global_avgpool')
        self.dense = layers.Dense(10, activation='softmax', name=f'{name_prefix}_logit')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.gap(x)
        x = self.dense(x)
        return x


class SharingWeightModel(models.Model):
    def __init__(self, space_sample):
        super(SharingWeightModel, self).__init__()
        self.s = ss()
        self.layers.append(self.s.bn)
        self.dd = layers.Dense(5)
        # self.layers.append(self.s.act)
        # self.layers.append(self.s.gap)
        # self.layers.append(self.s.dense)
        # self.conv = layers.Conv2D(
        #     filters=64,
        #     kernel_size=(3, 3),
        #     strides=(1, 1),
        #     padding='same',
        #     name=f'0_stem_conv2d'
        # )
        # self.bn = layers.BatchNormalization(name=f'0_stem_bn')
        # name_prefix = 'output'
        #
        # self.act = layers.Activation('relu', name=f'{name_prefix}_relu')
        # self.gap = layers.GlobalAveragePooling2D(name=f'{name_prefix}_global_avgpool')
        # self.dense = layers.Dense(10, activation='softmax', name=f'{name_prefix}_logit')
        # self.compiled_space = space_sample.compile(deepcopy=False)

    def update_search_space(self, space_sample):
        space_sample.compile(deepcopy=False)
        self.compiled_space = space_sample

    def call(self, inputs, training=None, mask=None):
        # x = self.conv(inputs)
        # x = self.bn(x)
        # x = self.act(x)
        # x = self.gap(x)
        # x = self.dense(x)
        # return x
        logits = self.s.call(inputs)
        return logits
        # logits = self.compiled_space.forward(inputs=inputs)
        # return logits


class KerasEstimator(Estimator):
    def __init__(self, space_sample, optimizer, loss, metrics, max_model_size=0, weights_cache=None,
                 visualization=False):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.max_model_size = max_model_size
        self.weights_cache = weights_cache
        self.visualization = visualization
        Estimator.__init__(self, space_sample=space_sample)

    def _build_model(self, space_sample):
        K.clear_session()
        gc.collect()
        space_sample.weights_cache = self.weights_cache
        model = space_sample.keras_model(deepcopy=self.weights_cache is None)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        if self.max_model_size > 0:
            model_size = compute_params_count(model)
            assert model_size <= self.max_model_size, f'Model size out of limit:{self.max_model_size}'
        if self.visualization:
            utils.plot_model(model, f'model_{space_sample.space_id}.png', show_shapes=True)
        return model

    def summary(self):
        self.model.summary()

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y, metrics=None, **kwargs):
        scores = self.model.evaluate(X, y, **kwargs)
        result = {k: v for k, v in zip(self.model.metrics_names, scores)}
        return result


class HyperKeras(HyperModel):
    def __init__(self, searcher, optimizer, loss, metrics, dispatcher=None, callbacks=[],
                 reward_metric=None, max_model_size=0, one_shot_mode=False, visualization=False):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.max_model_size = max_model_size
        if reward_metric is None:
            reward_metric = metrics[0]
        if one_shot_mode:
            self.weights_cache = LayerWeightsCache()
        else:
            self.weights_cache = None
        self.visualization = visualization
        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, reward_metric=reward_metric)

    def _get_estimator(self, space_sample):
        estimator = KerasEstimator(space_sample, optimizer=self.optimizer, loss=self.loss, metrics=self.metrics,
                                   max_model_size=self.max_model_size, weights_cache=self.weights_cache,
                                   visualization=self.visualization)
        return estimator

    def export_trail_configuration(self, trail):
        return None


def compute_params_count(model):
    assert model.built, ''
    return int(np.sum([K.count_params(weights) for weights in model.trainable_weights]))
