# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import time
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from collections import defaultdict


class BaseEstimator():
    def __init__(self, task=None):
        self.task = task
        self.name = None

    def train(self, X, y, X_test):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
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


class Evaluator():
    def evaluate(self, data, target, task, estimators, scorers, test_size=0.3, random_state=9527):
        if isinstance(data, tuple):
            assert len(data) == 2
            X_train = data[0]
            X_test = data[1]
            y_train = X_train.pop(target)
            y_test = X_test.pop(target)
        else:
            y = data.pop(target)
            if task == 'binary':
                stratify = y
            else:
                stratify = None
            X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, random_state=random_state,
                                                                stratify=stratify)

        result = []
        errors = []
        for estimator in estimators:
            try:
                starttime = time.time()
                estimator.train(X_train, y_train, X_test)
                elapsed = time.time() - starttime
                for scoring in scorers:
                    if isinstance(scoring, str):
                        scorer = get_scorer(scoring)
                        metric = scoring
                    else:
                        metric = str(scoring)
                    score = scorer(estimator, X_test, y_test) * scorer._sign
                    result.append((estimator.name, metric, score, elapsed, time.strftime("%Y-%m-%d %H:%M:%S")))
            except Exception as e:
                errors.append((estimator.name, str(e)))
                import traceback
                print(traceback.format_exc())

        return result, errors
