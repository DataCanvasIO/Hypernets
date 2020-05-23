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
    def __init__(self, space, optimizer, loss, metrics, max_model_size=0):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.max_model_size = max_model_size
        Estimator.__init__(self, space=space)

    def _build_model(self, space):
        K.clear_session()
        gc.collect()

        model = space.keras_model()
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
        self.model.predict(X, **kwargs)

    def evaluate(self, X, y, **kwargs):
        scores = self.model.evaluate(X, y, **kwargs)
        result = {k: v for k, v in zip(self.model.metrics_names, scores)}
        return result


class HyperKeras(HyperModel):
    def __init__(self, searcher, optimizer, loss, metrics, dispatcher=None, callbacks=[], max_trails=10,
                 reward_metric=None, max_model_size=0):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.max_model_size = max_model_size
        if reward_metric is None:
            reward_metric = metrics[0]
        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, max_trails=max_trails,
                            reward_metric=reward_metric)

    def _get_estimator(self, space):
        estimator = KerasEstimator(space, optimizer=self.optimizer, loss=self.loss, metrics=self.metrics,
                                   max_model_size=self.max_model_size)
        return estimator


def compute_params_count(model):
    assert model.built, ''
    return int(np.sum([K.count_params(weights) for weights in model.trainable_weights]))

#
# class KerasHyperModel(HyperModel):
#     """Builds and compiles a Keras Model with optional compile overrides."""
#
#     def __init__(self,
#                  hypermodel,
#                  max_model_size=None,
#                  optimizer=None,
#                  loss=None,
#                  metrics=None,
#                  distribution_strategy=None,
#                  **kwargs):
#         super(KerasHyperModel, self).__init__(**kwargs)
#         self.hypermodel = get_hypermodel(hypermodel)
#         self.max_model_size = max_model_size
#         self.optimizer = optimizer
#         self.loss = loss
#         self.metrics = metrics
#         self.distribution_strategy = distribution_strategy
#
#         self._max_fail_streak = 5
#
#     def build(self, hp):
#         for i in range(self._max_fail_streak + 1):
#             # clean-up TF graph from previously stored (defunct) graph
#             keras.backend.clear_session()
#             gc.collect()
#
#             # Build a model, allowing max_fail_streak failed attempts.
#             try:
#                 with maybe_distribute(self.distribution_strategy):
#                     model = self.hypermodel.build(hp)
#             except:
#                 if config_module.DEBUG:
#                     traceback.print_exc()
#
#                 display.warning('Invalid model %s/%s' %
#                                 (i, self._max_fail_streak))
#
#                 if i == self._max_fail_streak:
#                     raise RuntimeError(
#                         'Too many failed attempts to build model.')
#                 continue
#
#             # Stop if `build()` does not return a valid model.
#             if not isinstance(model, keras.models.Model):
#                 raise RuntimeError(
#                     'Model-building function did not return '
#                     'a valid Keras Model instance, found {}'.format(model))
#
#             # Check model size.
#             size = maybe_compute_model_size(model)
#             if self.max_model_size and size > self.max_model_size:
#                 display.warning(
#                     'Oversized model: %s parameters -- skipping' % (size))
#                 if i == self._max_fail_streak:
#                     raise RuntimeError(
#                         'Too many consecutive oversized models.')
#                 continue
#             break
#
#         return self._compile_model(model)
#
#     def _compile_model(self, model):
#         with maybe_distribute(self.distribution_strategy):
#             if self.optimizer or self.loss or self.metrics:
#                 compile_kwargs = {
#                     'optimizer': model.optimizer,
#                     'loss': model.loss,
#                     'metrics': model.metrics,
#                 }
#                 if self.loss:
#                     compile_kwargs['loss'] = self.loss
#                 if self.optimizer:
#                     compile_kwargs['optimizer'] = self.optimizer
#                 if self.metrics:
#                     compile_kwargs['metrics'] = self.metrics
#                 model.compile(**compile_kwargs)
#             return model
