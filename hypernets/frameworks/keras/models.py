# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.model.hyper_model import HyperModel
from hypernets.model.estimator import Estimator
from tensorflow.keras import backend as K
import numpy as np
import gc


class KerasEstimator(Estimator):
    def __init__(self, space_sample, optimizer, loss, metrics, max_model_size=0):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.max_model_size = max_model_size
        Estimator.__init__(self, space=space_sample)

    def _build_model(self, space_sample):
        K.clear_session()
        gc.collect()

        model = space_sample.keras_model()
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        if self.max_model_size > 0:
            model_size = compute_params_count(model)
            assert model_size <= self.max_model_size, f'Model size out of limit:{self.max_model_size}'

        return model

    def summary(self):
        self.model.summary()

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y, **kwargs):
        scores = self.model.evaluate(X, y, **kwargs)
        result = {k: v for k, v in zip(self.model.metrics_names, scores)}
        return result


class HyperKeras(HyperModel):
    def __init__(self, searcher, optimizer, loss, metrics, dispatcher=None, callbacks=[],
                 reward_metric=None, max_model_size=0):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.max_model_size = max_model_size
        if reward_metric is None:
            reward_metric = metrics[0]
        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, reward_metric=reward_metric)

    def _get_estimator(self, space_sample):
        estimator = KerasEstimator(space_sample, optimizer=self.optimizer, loss=self.loss, metrics=self.metrics,
                                   max_model_size=self.max_model_size)
        return estimator

    def export_trail_configuration(self):
        return None

def compute_params_count(model):
    assert model.built, ''
    return int(np.sum([K.count_params(weights) for weights in model.trainable_weights]))
