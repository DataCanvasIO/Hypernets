# -*- coding:utf-8 -*-
"""

"""
import copy

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


class Estimator():
    def __init__(self, space_sample, task='binary'):
        self.space_sample = space_sample
        self.task = task

    def summary(self):
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def fit_cross_validation(self, X, y, stratified=True, num_folds=3,
                             shuffle=False, random_state=9527, metrics=None):
        raise NotImplementedError

    def predict(self, X, **kwargs):
        raise NotImplementedError

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError

    def evaluate(self, X, y, metrics=None, **kwargs):
        raise NotImplementedError

    def save(self, model_file):
        raise NotImplementedError

    @staticmethod
    def load(model_file):
        raise NotImplementedError

    def proba2predict(self, proba, proba_threshold=0.5):
        if self.task == 'regression':
            return proba
        if proba.shape[-1] > 2:
            predict = proba.argmax(axis=-1)
        elif proba.shape[-1] == 2:
            predict = (proba[:, 1] > proba_threshold).astype('int32')
        else:
            predict = (proba > proba_threshold).astype('int32')
        return predict


class CrossValidationEstimator():
    def __init__(self, base_estimator, task, num_folds=3, stratified=False, shuffle=False, random_state=None):
        self.base_estimator = base_estimator
        self.num_folds = num_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        self.task = task
        self.oof_ = None
        self.classes_ = None
        self.estimators_ = []

    def fit(self, X, y, **kwargs):
        self.oof_ = None
        self.estimators_ = []
        if self.stratified and self.task == 'binary':
            iterators = StratifiedKFold(n_splits=self.num_folds, shuffle=self.shuffle, random_state=self.random_state)
        else:
            iterators = KFold(n_splits=self.num_folds, shuffle=self.shuffle, random_state=self.random_state)

        y = np.array(y)
        sample_weight = kwargs.get('sample_weight')

        for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X, y)):
            x_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
            x_val_fold, y_val_fold = X.iloc[valid_idx], y[valid_idx]

            kwargs['eval_set'] = [(x_val_fold, y_val_fold)]
            if sample_weight is not None:
                sw_fold = sample_weight[train_idx]
                kwargs['sample_weight'] = sw_fold
            fold_est = copy.deepcopy(self.base_estimator)
            fold_est.fit(x_train_fold, y_train_fold, **kwargs)
            if self.classes_ is None:
                self.classes_ = fold_est.classes_
            if self.task == 'regression':
                proba = fold_est.predict(x_val_fold)
            else:
                proba = fold_est.predict_proba(x_val_fold)

            if self.oof_ is None:
                if len(proba.shape) == 1:
                    self.oof_ = np.zeros(y.shape, proba.dtype)
                else:
                    self.oof_ = np.zeros((y.shape[0], proba.shape[-1]), proba.dtype)
            self.oof_[valid_idx] = proba
            self.estimators_.append(fold_est)

        return self

    def predict_proba(self, X):
        proba_sum = None
        for est in self.estimators_:
            proba = est.predict_proba(X)
            if proba_sum is None:
                proba_sum = np.zeros_like(proba)
            proba_sum += proba
        return proba_sum / len(self.estimators_)

    def predict(self, X):
        if self.task == 'regression':
            pred_sum = None
            for est in self.estimators_:
                pred = est.predict(X)
                if pred_sum is None:
                    pred_sum = np.zeros_like(pred)
                pred_sum += pred
            return pred_sum / len(self.estimators_)
        elif self.task == 'binary':
            proba = self.predict_proba(X)
            pred = self.proba2predict(proba)
            pred = np.array(self.classes_).take(pred, axis=0)
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
