# -*- coding:utf-8 -*-
"""

"""
import time


class Estimator():
    def __init__(self, space_sample, task='classification'):
        self.space_sample = space_sample
        self.task = task

    def summary(self):
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
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
        if self.task != 'classification':
            return proba
        if proba.shape[-1] > 2:
            predict = proba.argmax(axis=-1)
        elif proba.shape[-1] == 2:
            predict = (proba[:, 1] > proba_threshold).astype('int32')
        else:
            predict = (proba > proba_threshold).astype('int32')
        return predict
