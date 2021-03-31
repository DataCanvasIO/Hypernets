# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from . import BaseEstimator
import h2o
from h2o.automl import H2OAutoML


class H2OEstimator(BaseEstimator):
    def __init__(self, task, **kwargs):
        super(H2OEstimator, self).__init__(task)
        self.name = 'H2O AutoML'
        self.kwargs = kwargs
        self.estimator = None

    def train(self, X, y, X_test):
        h2o.init()
        target = '__tabular_toolbox_target__'
        X.insert(0, target, y)
        train = h2o.H2OFrame(X)
        x_cols = train.columns
        x_cols.remove(target)
        train[target] = train[target].asfactor()
        self.esitmator = H2OAutoML(max_models=20, seed=1)
        self.esitmator.train(x=x_cols, y=target, training_frame=train)

    def predict_proba(self, X):
        x = h2o.H2OFrame(X)
        preds = self.esitmator.predict(x)
        preds = preds[1:].as_data_frame().values
        return preds

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.proba2predict(proba)
