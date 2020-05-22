# -*- coding:utf-8 -*-
"""

"""


class Estimator():
    def __init__(self, space):
        self.space = space
        self.model = self._build_model(space)

    def _build_model(self, space):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X, **kwargs):
        raise NotImplementedError

    def evaluate(self, X, y, **kwargs):
        raise NotImplementedError
