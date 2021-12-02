# -*- coding:utf-8 -*-
"""

"""
import cuml
from hypernets.utils import logging

logger = logging.get_logger(__name__)


def detect_xgboost():
    try:
        import xgboost
    except ImportError:
        return False

    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=50, n_features=5, n_classes=2)
    xgb = xgboost.XGBClassifier(tree_method='gpu_hist', use_label_encoder=False)
    try:
        xgb.fit(X, y, eval_metric='logloss')
        return True
    except xgboost.core.XGBoostError as e:
        logger.warn(e)

    return False


def detect_lightgbm():
    try:
        import lightgbm
    except ImportError:
        return False

    from sklearn.datasets import make_classification
    import pandas as pd
    X, y = make_classification(n_samples=50, n_features=5, n_classes=2)
    X = pd.DataFrame(X, columns=[f'c{i}' for i in range(X.shape[1])])
    try:
        lgbm = lightgbm.LGBMClassifier(device='GPU')
        lgbm.fit(X, y)
        return True
    except Exception as e:
        if str(e).find('USE_GPU') > 0:
            logger.warn(f'your lightgbm does not support GPU devices. details:\n{type(e).__name__}: {str(e)}')
        else:
            logger.warn(e)

    return False


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
