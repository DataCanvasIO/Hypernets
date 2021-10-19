# -*- coding:utf-8 -*-
"""

"""
import dask.array as da
import dask.dataframe as dd
from sklearn import model_selection as sk_sel


def _fake_X_y(X, y):
    if isinstance(X, dd.DataFrame):
        X = X.index.to_frame()
        X.set_index(0)
        X = X.compute()

    if isinstance(y, (dd.Series, dd.DataFrame, da.Array)):
        y = y.compute()

    return X, y


class FakeDaskKFold(sk_sel.KFold):
    def split(self, X, y=None, groups=None):
        X, y = _fake_X_y(X, y)
        yield from super().split(X, y, groups=groups)


class FakeDaskStratifiedKFold(sk_sel.StratifiedKFold):
    def split(self, X, y, groups=None):
        assert y is not None

        X, y = _fake_X_y(X, y)
        yield from super().split(X, y, groups=groups)
