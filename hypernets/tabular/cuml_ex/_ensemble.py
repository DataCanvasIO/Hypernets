# -*- coding:utf-8 -*-
"""

"""
import cudf
import cupy

from hypernets.tabular.ensemble import GreedyEnsemble
from ._transformer import Localizable, as_local_if_possible, copy_attrs_as_local


class CumlGreedyEnsemble(GreedyEnsemble, Localizable):
    np = cupy

    @staticmethod
    def _to_local(y):
        if isinstance(y, cupy.ndarray):
            y = cupy.asnumpy(y)
        elif isinstance(y, cudf.Series):
            y = y.to_pandas()

        return y

    def _score(self, y_true, y_preds):
        y_true = self._to_local(y_true)
        y_preds = list(map(self._to_local, y_preds))

        r = super()._score(y_true, y_preds)
        return r

    def as_local(self):
        estimators = list(map(as_local_if_possible, self.estimators))
        target = GreedyEnsemble(estimators=estimators, task=self.task, need_fit=self.need_fit,
                                n_folds=self.n_folds, method=self.method, random_state=self.random_state,
                                scoring=self.scoring, ensemble_size=self.ensemble_size)
        copy_attrs_as_local(self, target, 'weights_', 'scores_', 'hits_', 'best_stack_')
        return target
