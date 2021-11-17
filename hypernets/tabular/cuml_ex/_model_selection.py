# -*- coding:utf-8 -*-
"""

"""
import cudf
import cupy
import numpy as np
from sklearn import model_selection as sk_sel
from cuml import model_selection as cm_sel


def train_test_split(*data, shuffle=True, random_state=None, stratify=None, **kwargs):
    return cm_sel.train_test_split(*data, shuffle=shuffle, random_state=random_state, stratify=stratify, **kwargs)


class FakeKFold(sk_sel.KFold):
    def split(self, X, y=None, groups=None):
        assert y is None or len(X) == len(y)
        X = y = np.arange(len(X))

        yield from super().split(X, y, groups=groups)


class FakeStratifiedKFold(sk_sel.StratifiedKFold):
    def split(self, X, y, groups=None):
        assert y is not None

        X = np.arange(len(X))

        if isinstance(y, (cudf.Series, cudf.DataFrame)):
            y = y.to_pandas()
        elif isinstance(y, cupy.ndarray):
            y = cupy.asnumpy(y)

        yield from super().split(X, y, groups=groups)
