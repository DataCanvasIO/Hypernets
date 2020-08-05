# -*- coding:utf-8 -*-
"""

"""


class Estimator():
    def __init__(self, space_sample):
        self.space_sample = space_sample
        self.model = self._build_model(space_sample)

    def _build_model(self, space_sample):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X, **kwargs):
        raise NotImplementedError

    def evaluate(self, X, y, metrics=None, **kwargs):
        raise NotImplementedError
