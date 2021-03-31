# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from .base_ensemble import BaseEnsemble
from sklearn.linear_model import LogisticRegression, LinearRegression

import numpy as np


class StackingEnsemble(BaseEnsemble):
    def __init__(self, task, estimators, need_fit=False, n_folds=5, method='soft', meta_model=None, fit_kwargs=None):
        super(StackingEnsemble, self).__init__(task, estimators, need_fit, n_folds, method)
        if meta_model is None:
            if task == 'regression':
                self.meta_model = LinearRegression()
            else:
                self.meta_model = LogisticRegression()
        else:
            self.meta_model = meta_model
        self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}

    def fit_predictions(self, predictions, y_true):
        X = self.__predictions2X(predictions)
        self.meta_model.fit(X, y_true, **self.fit_kwargs)

    def __predictions2X(self, predictions):
        X = predictions
        if len(X.shape) == 3:
            if self.task == 'binary':
                X = X[:, :, -1]
            elif self.task == 'multiclass':
                X = np.argmax(X, axis=2)
            else:
                raise ValueError(
                    f"The shape of `predictions` and the `task` don't match. shape:{predictions.shape}, task:{self.task}")
        return X

    def predictions2predict(self, predictions):
        assert self.meta_model is not None
        X = self.__predictions2X(predictions)
        pred = self.meta_model.predict(X)
        if self.task == 'binary':
            pred = np.clip(pred, 0, 1)
        return pred

    def predictions2predict_proba(self, predictions):
        assert self.meta_model is not None
        X = self.__predictions2X(predictions)
        if hasattr(self.meta_model, 'predict_proba'):
            pred = self.meta_model.predict_proba(X)
        else:
            pred = self.meta_model.predict(X)

        if self.task == 'binary':
            pred = np.clip(pred, 0, 1)
        return pred
