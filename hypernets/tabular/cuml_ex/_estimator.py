# -*- coding:utf-8 -*-
"""

"""
import cuml

from hypernets.utils import logging
from ._transformer import LocalizableLabelEncoder as LabelEncoder

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

        def fit(self, X, y, **kwargs):
            from .. import CumlToolBox

            le = LabelEncoder()
            y = le.fit_transform(y)
            le.classes_, = CumlToolBox.to_local(le.classes_)
            super().fit(X, y, **kwargs)
            self.y_encoder_ = le
            return self

        def predict(self, X, **kwargs):
            from .. import CumlToolBox
            pred = super().predict(X, **kwargs)
            pred, = CumlToolBox.from_local(pred, enable_cuml_array=False)
            pred = self.y_encoder_.inverse_transform(pred)
            return pred

        def predict_proba(self, X, **kwargs):
            from .. import CumlToolBox
            proba = super().predict_proba(X, **kwargs)
            proba, = CumlToolBox.from_local(proba, enable_cuml_array=False)
            return proba

        def __getattribute__(self, name):
            if name == 'classes_':
                le = getattr(self, 'y_encoder_', None)
                if le is not None:
                    return le.classes_

            return super().__getattribute__(name)


    class AdaptedXGBRegressor(xgboost.XGBRegressor):
        def predict(self, X, **kwargs):
            from .. import CumlToolBox
            pred = super().predict(X, **kwargs)
            pred, = CumlToolBox.from_local(pred, enable_cuml_array=False)
            return pred

except ImportError:
    pass
