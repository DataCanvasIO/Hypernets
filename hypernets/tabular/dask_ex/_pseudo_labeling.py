# -*- coding:utf-8 -*-
"""

"""

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np

from hypernets.utils import logging
from ..pseudo_labeling import PseudoLabeling

logger = logging.get_logger(__name__)


# def _select_top(chunk, number):
#     pos = chunk.shape[0] - number
#     i = np.argsort(np.argsort(chunk, axis=0), axis=0)
#     result = np.logical_and(i >= pos, chunk > 0)
#     return result
#

class DaskPseudoLabeling(PseudoLabeling):
    # def select(self, X_test, classes, proba):
    #     assert len(classes) == proba.shape[-1] > 1
    #     from ._toolbox import DaskToolBox
    #
    #     mx = proba.max(axis=1, keepdims=True)
    #     proba = da.where(proba < mx, 0, proba)
    #     proba = DaskToolBox.make_chunk_size_known(proba)
    #
    #     if self.strategy is None or self.strategy == DaskToolBox.STRATEGY_THRESHOLD:
    #         selected = (proba >= self.threshold)
    #     elif self.strategy == DaskToolBox.STRATEGY_NUMBER:
    #         if isinstance(self.number, float) and 0 < self.number < 1:
    #             proba_shape = dask.compute(proba.shape)[0] if DaskToolBox.is_dask_object(proba) else proba.shape
    #             number = int(proba_shape[0] / proba_shape[1] * self.number)
    #             if number < 10:
    #                 number = 10
    #         else:
    #             number = int(self.number)
    #         if proba.numblocks[0] > 1:
    #             proba = proba.rechunk(proba.shape)
    #         selected = proba.map_blocks(_select_top, number, meta=np.array((), 'bool'))
    #     elif self.strategy == DaskToolBox.STRATEGY_QUANTILE:
    #         qs = []
    #         for i in range(proba.shape[-1]):
    #             c = proba[:, i]
    #             ci = da.argwhere(c > 0)
    #             ci = DaskToolBox.make_chunk_size_known(ci).ravel()
    #             t = c[ci]
    #             qs.append(da.percentile(t, self.quantile * 100))
    #         qs = dask.compute(qs)[0]
    #         selected = (proba >= np.array(qs).ravel())
    #     else:
    #         raise ValueError(f'Unsupported strategy: {self.strategy}')
    #
    #     pred = (selected * np.arange(1, len(classes) + 1)).max(axis=1) - 1
    #     idx = da.argwhere(pred >= 0)
    #     idx = DaskToolBox.make_chunk_size_known(idx).ravel()
    #
    #     if DaskToolBox.is_dask_dataframe(X_test):
    #         X_test_values = X_test.to_dask_array(lengths=True)
    #         X_test_values = X_test_values[idx]
    #         X_pseudo = dd.from_dask_array(X_test_values, columns=X_test.columns,
    #                                       meta=X_test._meta)
    #     else:
    #         X_pseudo = X_test[idx]
    #     y_pseudo = da.take(np.array(classes), pred[idx], axis=0)
    #
    #     if logger.is_info_enabled():
    #         if DaskToolBox.is_dask_object(y_pseudo):
    #             y = dask.compute(y_pseudo)[0]
    #         else:
    #             y = y_pseudo
    #
    #         uniques = np.unique(y, return_counts=True)
    #         value_counts = {k: n for k, n in zip(uniques[0], uniques[1])}
    #         logger.info(f'extract pseudo labeling samples (strategy={self.strategy}): {value_counts}')
    #
    #     return X_pseudo, y_pseudo
    def select(self, X_test, classes, proba):
        from ._toolbox import DaskToolBox
        proba, = DaskToolBox.to_local(proba)

        X_pseudo, y_pseudo = super().select(X_test, classes, proba)
        y_pseudo, = DaskToolBox.from_local(y_pseudo)

        return X_pseudo, y_pseudo
