# -*- coding:utf-8 -*-
"""

"""
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array
import numpy as np


class SafeLabelEncoder(LabelEncoder):
    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        unseen = len(self.classes_)
        y = np.array([np.searchsorted(self.classes_, x) if x in self.classes_ else unseen for x in y])
        return y


class MultiLabelEncoder:
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        assert len(X.shape) == 2
        n_features = X.shape[1]
        for n in range(n_features):
            le = SafeLabelEncoder()
            le.fit(X[:, n])
            self.encoders[n] = le
        return self

    def transform(self, X):
        assert len(X.shape) == 2
        n_features = X.shape[1]
        assert n_features == len(self.encoders.items())
        for n in range(n_features):
            X[:, n] = self.encoders[n].transform(X[:, n])
        return X
