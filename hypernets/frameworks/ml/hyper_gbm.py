# -*- coding:utf-8 -*-
"""

"""
from hypernets.model.hyper_model import HyperModel
from hypernets.model.estimator import Estimator
import numpy as np
import lightgbm
from .preprocessing import *
from .estimators import HyperEstimator


class HyperGBMModel():
    def __init__(self, data_pipeline, estimator, fit_kwargs=None):
        self.data_pipeline = data_pipeline
        self.estimator = estimator
        self.fit_kwargs = fit_kwargs

    def fit(self, X, y, **kwargs):
        X = self.data_pipeline.fit_transform(X, y)
        if self.fit_kwargs is not None:
            kwargs = self.fit_kwargs
        self.estimator.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        X = self.data_pipeline.transform(X)
        self.estimator.predict(X, **kwargs)


class HyperGBMEstimator(Estimator):
    def __init__(self, space_sample):
        Estimator.__init__(self, space=space_sample)

    def _build_model(self, space_sample):
        space = space_sample.compile_space()

        outputs = space.get_outputs()
        assert len(outputs) == 1, 'The space can only contains 1 output.'
        assert isinstance(outputs[0], HyperEstimator), 'The output of space must be `HyperEstimator`.'
        estimator = outputs[0].estimator
        fit_kwargs = outputs[0].fit_args

        pipeline_module = space.get_inputs(outputs[0])
        assert len(pipeline_module) == 1, 'The `HyperEstimator` can only contains 1 input.'
        assert isinstance(pipeline_module[0],
                          ComposeTransformer), 'The upstream node of `HyperEstimator` must be `ComposeTransformer`.'
        next, (name, p) = pipeline_module[0].compose()
        model = HyperGBMModel(p, estimator, fit_kwargs)
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
