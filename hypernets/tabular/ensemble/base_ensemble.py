# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import numpy as np
from sklearn.model_selection import StratifiedKFold


class BaseEnsemble():
    def __init__(self, task, estimators, need_fit=False, n_folds=5, method='soft', random_state=9527):
        self.task = task
        self.estimators = list(estimators)
        self.need_fit = need_fit
        self.method = method
        self.n_folds = n_folds
        self.random_state = random_state
        self.classes_ = None
        self.preds_shape_ = None
        for est in estimators:
            if est is not None and self.classes_ is None and hasattr(est, 'classes_'):
                self.classes_ = est.classes_
                break

    def __predict(self, estimator, X):
        if self.task == 'regression':
            pred = estimator.predict(X)
        else:
            if self.classes_ is None and hasattr(estimator, 'classes_'):
                self.classes_ = estimator.classes_
            pred = estimator.predict_proba(X)
            if self.method == 'hard':
                pred = self.proba2predict(pred)
        return pred

    def proba2predict(self, proba, proba_threshold=0.5):
        assert len(proba.shape) <= 2
        if self.task == 'regression':
            return proba
        if len(proba.shape) == 2:
            if proba.shape[-1] > 2:
                predict = proba.argmax(axis=-1)
            else:
                predict = (proba[:, -1] > proba_threshold).astype('int32')
        else:
            predict = (proba > proba_threshold).astype('int32')

        return predict

    def fit(self, X, y, est_predictions=None):
        assert y is not None
        if est_predictions is not None:
            # print(f'est_predictions.shape:{est_predictions.shape}, estimators:{len(self.estimators)}')
            if self.task == 'regression' or self.method == 'hard':
                assert est_predictions.shape == (len(y), len(
                    self.estimators)), f'shape is not equal, may be a wrong task type. task:{self.task},  est_predictions.shape: {est_predictions.shape}, (len(y), len(self.estimators)):{(len(y), len(self.estimators))}'
            else:
                assert len(est_predictions.shape) == 3
                assert est_predictions.shape[0] == len(y)
                assert est_predictions.shape[1] == len(self.estimators)
        else:
            assert X is not None
            if self.task == 'regression' or self.method == 'hard':
                est_predictions = np.zeros((len(y), len(self.estimators)), dtype=np.float64)
            if self.need_fit:
                iterators = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                for fold, (train, test) in enumerate(iterators.split(X, y)):
                    for n, estimator in enumerate(self.estimators):
                        X_train = X.iloc[train]
                        y_train = y.iloc[train]
                        X_test = X.iloc[test]
                        estimator.fit(X_train, y_train)
                        pred = self.__predict(estimator, X_test)
                        if est_predictions is None:
                            est_predictions = np.zeros((len(y), len(self.estimators), pred.shape[1]), dtype=np.float64)
                        est_predictions[test, n] = pred

            else:
                for n, estimator in enumerate(self.estimators):
                    pred = self.__predict(estimator, X)
                    if est_predictions is None:
                        est_predictions = np.zeros((len(y), len(self.estimators), pred.shape[1]), dtype=np.float64)
                    est_predictions[:, n] = pred

        self.preds_shape_ = est_predictions.shape
        self.fit_predictions(est_predictions, y)

    def X2predictions(self, X):
        if self.task == 'regression' or self.method == 'hard':
            est_predictions = np.zeros((len(X), len(self.estimators)), dtype=np.float64)
        else:
            est_predictions = np.zeros((len(X), len(self.estimators), self.preds_shape_[-1]), dtype=np.float64)
        for n, estimator in enumerate(self.estimators):
            if estimator is not None:
                pred = self.__predict(estimator, X)
                est_predictions[:, n] = pred
        return est_predictions

    def predict(self, X):
        est_predictions = self.X2predictions(X)
        pred = self.predictions2predict(est_predictions)
        if self.task != 'regression' and self.classes_ is not None:
            pred = np.array(self.classes_).take(pred, axis=0)
        return pred

    def predict_proba(self, X):
        est_predictions = self.X2predictions(X)
        return self.predictions2predict_proba(est_predictions)

    def fit_predictions(self, predictions, y_true):
        raise NotImplementedError()

    def predictions2predict_proba(self, predictions):
        raise NotImplementedError()

    def predictions2predict(self, predictions):
        raise NotImplementedError()
