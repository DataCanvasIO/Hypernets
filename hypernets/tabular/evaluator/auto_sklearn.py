# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import autosklearn.classification
import autosklearn.regression
from . import BaseEstimator
from ..column_selector import column_object


class AutoSklearnEstimator(BaseEstimator):
    def __init__(self, task, **kwargs):
        super(AutoSklearnEstimator, self).__init__(task)
        if task == 'regression':
            self.automl = autosklearn.regression.AutoSklearnRegressor(**kwargs)
        else:
            self.automl = autosklearn.classification.AutoSklearnClassifier(**kwargs)
        self.name = 'auto-sklearn'

    def train(self, X, y, X_test):
        target = '__tabular_toolbox_target__'
        X.insert(0, target, y)
        obj_cols = column_object(X)
        if len(obj_cols) > 0:
            X[obj_cols] = X[obj_cols].astype('category')
        y = X.pop(target)
        self.automl.fit(X, y)

    def predict_proba(self, X):
        obj_cols = column_object(X)
        if len(obj_cols) > 0:
            X[obj_cols] = X[obj_cols].astype('category')
        return self.automl.predict_proba(X)

    def predict(self, X):
        obj_cols = column_object(X)
        if len(obj_cols) > 0:
            X[obj_cols] = X[obj_cols].astype('category')
        return self.automl.predict(X)
