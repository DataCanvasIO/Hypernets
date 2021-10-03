# -*- coding:utf-8 -*-
"""

"""

import dask

from ._transformers import SafeOrdinalEncoder
from ..collinearity import MultiCollinearityDetector


class DaskMultiCollinearityDetector(MultiCollinearityDetector):
    def _value_counts(self, X):
        n_values = super()._value_counts(X)
        return dask.compute(*n_values)

    def _corr(self, X, method=None):
        Xt = SafeOrdinalEncoder().fit_transform(X)
        corr = Xt.corr(method='pearson' if method is None else method).compute().values
        return corr
