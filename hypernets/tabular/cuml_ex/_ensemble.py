# -*- coding:utf-8 -*-
"""

"""
import cudf
import cupy

from hypernets.tabular.ensemble import GreedyEnsemble
from ._transformer import Localizable, as_local_if_possible, copy_attrs_as_local


class CumlGreedyEnsemble(GreedyEnsemble, Localizable):
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

    def as_local(self):
        estimators = list(map(as_local_if_possible, self.estimators))
        target = GreedyEnsemble(estimators=estimators, task=self.task, need_fit=self.need_fit,
                                n_folds=self.n_folds, method=self.method, random_state=self.random_state,
                                scoring=self.scoring, ensemble_size=self.ensemble_size)
        copy_attrs_as_local(self, target, 'weights_', 'scores_', 'hits_', 'best_stack_')
        return target
