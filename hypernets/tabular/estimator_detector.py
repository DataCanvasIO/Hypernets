# -*- coding:utf-8 -*-
"""

"""

import pandas as pd
from sklearn.datasets import make_classification, make_regression

from hypernets.core import randint
from hypernets.utils import logging, load_module, const

logger = logging.get_logger(__name__)


class EstimatorDetector:
    def __init__(self, name_or_cls, task, *,
                 init_kwargs=None, fit_kwargs=None, n_samples=100, n_features=5):
        assert isinstance(name_or_cls, (str, type))

        if init_kwargs is None:
            init_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}

        self.name_or_cls = name_or_cls
        self.task = task
        self.init_kwargs = init_kwargs
        self.fit_kwargs = fit_kwargs
        self.n_samples = n_samples
        self.n_features = n_features

    def prepare_data(self):
        if self.task == const.TASK_BINARY:
            X, y = make_classification(n_samples=self.n_samples, n_features=self.n_features, n_classes=2,
                                       random_state=randint())
        elif self.task == const.TASK_MULTICLASS:
            X, y = make_classification(n_samples=self.n_samples, n_features=self.n_features, n_classes=5,
                                       random_state=randint())
        else:
            X, y = make_regression(n_samples=self.n_samples, n_features=self.n_features,
                                   random_state=randint())
        X = pd.DataFrame(X, columns=[f'c{i}' for i in range(X.shape[1])])

        return X, y

    def get_estimator_cls(self):
        if isinstance(self.name_or_cls, str):
            estimator_cls = load_module(self.name_or_cls)
        else:
            estimator_cls = self.name_or_cls
        return estimator_cls

    def create_estimator(self, estimator_cls):
        return estimator_cls(**self.init_kwargs)

    def fit_estimator(self, estimator, X, y):
        return estimator.fit(X, y, **self.fit_kwargs)

    def __call__(self, *args, **kwargs):
        result = set([])

        # detect: installed
        try:
            estimator_cls = self.get_estimator_cls()
            result.add('installed')
        except ImportError:
            return result

        # detect: create estimator instance
        try:
            estimator = self.create_estimator(estimator_cls)
            result.add('initialized')
        except Exception as e:
            logger.info(e)
            return result

        # make training data
        X, y = self.prepare_data()

        # detect: fit
        try:
            self.fit_estimator(estimator, X, y)
            result.add('fitted')
        except Exception as e:
            logger.info(f'EstimatorDetector error: {e}, ')
            # logger.info(e)
            if logger.is_debug_enabled():
                logger.debug(e)

        return result
