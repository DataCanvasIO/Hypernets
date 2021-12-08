# -*- coding:utf-8 -*-
"""

"""
import cuml

from hypernets.utils import logging

logger = logging.get_logger(__name__)


class AdaptedRandomForestClassifier(cuml.ensemble.RandomForestClassifier):
    def fit(self, X, y, *args, **kwargs):
        X = X.astype('float32')
        y = y.astype('float32')

        return super().fit(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        X = X.astype('float32')
        return super().predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        X = X.astype('float32')
        return super().predict_proba(X, *args, **kwargs)


class AdaptedRandomForestRegressor(cuml.ensemble.RandomForestRegressor):
    def fit(self, X, y, *args, **kwargs):
        X = X.astype('float32')
        y = y.astype('float32')

        return super().fit(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        X = X.astype('float32')
        return super().predict(X, *args, **kwargs)


try:
    import xgboost


    class AdaptedXGBClassifier(xgboost.XGBClassifier):
        def predict(self, X, **kwargs):
            from .. import CumlToolBox
            pred = super().predict(X, **kwargs)
            pred, = CumlToolBox.from_local(pred, enable_cuml_array=False)
            return pred

        def predict_proba(self, X, **kwargs):
            from .. import CumlToolBox
            proba = super().predict_proba(X, **kwargs)
            proba, = CumlToolBox.from_local(proba, enable_cuml_array=False)
            return proba


    class AdaptedXGBRegressor(xgboost.XGBRegressor):
        def predict(self, X, **kwargs):
            from .. import CumlToolBox
            pred = super().predict(X, **kwargs)
            pred, = CumlToolBox.from_local(pred, enable_cuml_array=False)
            return pred

except ImportError:
    pass
