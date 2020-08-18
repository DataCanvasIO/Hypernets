# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.model.hyper_model import HyperModel
from hypernets.model.estimator import Estimator

from tensorflow.keras import backend as K
from tensorflow.keras import utils
import tensorflow as tf
import numpy as np
import gc
from .layer_weights_cache import LayerWeightsCache
import logging

logger = logging.getLogger(__name__)


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
                 reward_metric=None, max_model_size=0, one_shot_mode=False, one_shot_train_sampler=None,
                 visualization=False):
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
        self.one_shot_mode = one_shot_mode
        self.one_shot_train_sampler = one_shot_train_sampler if one_shot_train_sampler is not None else searcher
        self.visualization = visualization
        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, reward_metric=reward_metric)

    def _get_estimator(self, space_sample):
        estimator = KerasEstimator(space_sample, optimizer=self.optimizer, loss=self.loss, metrics=self.metrics,
                                   max_model_size=self.max_model_size, weights_cache=self.weights_cache,
                                   visualization=self.visualization)
        return estimator

    def build_dataset_iter(self, X, y, batch_size=32, buffer_size=None, reshuffle_each_iteration=None, repeat_count=1):
        if buffer_size is None:
            buffer_size = len(X[-1])
        dataset = tf.data.Dataset. \
            from_tensor_slices((X, y)). \
            shuffle(buffer_size=buffer_size, reshuffle_each_iteration=reshuffle_each_iteration). \
            repeat(repeat_count). \
            batch(batch_size)

        return iter(dataset)

    def fit_one_shot_model_epoch(self, X, y, batch_size=32, steps=None, epoch=0):
        step = 0
        dataset_iter = self.build_dataset_iter(X, y, batch_size=batch_size)
        for X_batch, y_batch in dataset_iter:
            sample = self.one_shot_train_sampler.sample()
            est = self._get_estimator(space_sample=sample)
            est.fit(X_batch, y_batch, batch_size=batch_size, epochs=1)
            step += 1
            print(f'One-shot model training, Epoch[{epoch}], Step[{step}]')
            if steps is not None and step >= steps:
                break
        print(f'One-shot model training finished. Epoch[{epoch}], Step[{step}]')

    def export_trail_configuration(self, trail):
        return None


def compute_params_count(model):
    assert model.built, ''
    return int(np.sum([K.count_params(weights) for weights in model.trainable_weights]))
