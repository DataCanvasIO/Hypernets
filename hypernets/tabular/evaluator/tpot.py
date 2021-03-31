# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from tpot import TPOTClassifier, TPOTRegressor
from . import BaseEstimator
from ..column_selector import column_object_category_bool
from ..sklearn_ex import SafeOrdinalEncoder


class TpotEstimator(BaseEstimator):
    def __init__(self, task, **kwargs):
        super(TpotEstimator, self).__init__(task)
        if task == 'regression':
            self.tpot = TPOTRegressor(**kwargs)
        else:
            self.tpot = TPOTClassifier(**kwargs)
        self.name = 'tpot'
        self.label_encoder = None
        self.obj_cols = None

    def train(self, X, y, X_test):
        self.obj_cols = column_object_category_bool(X)
        self.label_encoder = SafeOrdinalEncoder()
        X[self.obj_cols] = self.label_encoder.fit_transform(X[self.obj_cols])
        self.tpot.fit(X, y)

    def predict_proba(self, X):
        X[self.obj_cols] = self.label_encoder.transform(X[self.obj_cols])
        proba = self.tpot.predict_proba(X)
        print(f'proba.shape:{proba.shape}')
        return proba

    def predict(self, X):
        X[self.obj_cols] = self.label_encoder.transform(X[self.obj_cols])
        return self.tpot.predict(X)
