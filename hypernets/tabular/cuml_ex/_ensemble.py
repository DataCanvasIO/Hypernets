# -*- coding:utf-8 -*-
"""

"""
import cupy
import cudf
from hypernets.tabular.ensemble import GreedyEnsemble


class CumlGreedyEnsemble(GreedyEnsemble):
    np = cupy

    def _score(self, y_true, y_pred):
        if isinstance(y_true, cupy.ndarray):
            y_true = cupy.asnumpy(y_true)
        elif isinstance(y_true, cudf.Series):
            y_true = y_true.to_pandas()

        if isinstance(y_pred, cupy.ndarray):
            y_pred = cupy.asnumpy(y_pred)
        elif isinstance(y_pred, cudf.Series):
            y_pred = y_pred.to_pandas()

        s = super(CumlGreedyEnsemble, self)._score(y_true, y_pred)
        return s
