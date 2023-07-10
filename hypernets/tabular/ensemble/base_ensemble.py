# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import copy
import pickle

from sklearn.model_selection import StratifiedKFold

from hypernets.utils import fs, logging

logger = logging.get_logger(__name__)


class BaseEnsemble:
    import numpy as np

    def __init__(self, task, estimators, need_fit=False, n_folds=5, method='soft', random_state=9527):
        self.task = task
        self.estimators = list(estimators)
        self.need_fit = need_fit
        self.method = method
        self.n_folds = n_folds
        self.random_state = random_state
        self.classes_ = None
        for est in estimators:
            if est is not None and self.classes_ is None and hasattr(est, 'classes_'):
                self.classes_ = est.classes_
                break

    @property
    def _estimator_type(self):
        for est in self.estimators:
            if est is not None:
                return est._estimator_type
        return None

    def _estimator_predict(self, estimator, X):
        if self.task == 'regression':
            pred = estimator.predict(X)
        else:
            # if self.classes_ is None and hasattr(estimator, 'classes_'):
            #     self.classes_ = estimator.classes_
            assert self.classes_ is not None
            pred = estimator.predict_proba(X)
            if self.method == 'hard':
                pred = self.proba2predict(pred)
        return pred

    def _cross_validator(self):
        return StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

    def proba2predict(self, proba, proba_threshold=0.5):
        assert len(proba.shape) <= 2
        if self.task == 'regression':
            return proba
        if len(proba.shape) == 2:
            if proba.shape[-1] > 2:
                predict = proba.argmax(axis=-1)
            else:
                predict = (proba[:, -1] > proba_threshold).astype('int32')
        else:
            predict = (proba > proba_threshold).astype('int32')

        return predict

    def fit(self, X, y, est_predictions=None):
        assert y is not None
        if est_predictions is not None:
            logger.info('validate oof predictions')
            self._validate_predictions(X, y, est_predictions)
        else:
            assert X is not None
            if self.need_fit:
                logger.info(f'get predictions, need_fit={self.need_fit}')
                est_predictions = self._Xy2predicttions(X, y)
            else:
                logger.info(f'get predictions, need_fit={self.need_fit}')
                est_predictions = self._X2predictions(X)

        logger.info('fit_predictions')
        self.fit_predictions(est_predictions, y)

    def _validate_predictions(self, X, y, est_predictions):
        # print(f'est_predictions.shape:{est_predictions.shape}, estimators:{len(self.estimators)}')
        if self.task == 'regression' or self.method == 'hard':
            assert est_predictions.shape == (len(y), len(self.estimators)), \
                f'shape is not equal, may be a wrong task type. task:{self.task},  ' \
                f'est_predictions.shape: {est_predictions.shape}, ' \
                f'(len(y), len(self.estimators)):{(len(y), len(self.estimators))}'
        else:
            assert len(est_predictions.shape) == 3
            assert est_predictions.shape[0] == len(y)
            assert est_predictions.shape[1] == len(self.estimators)

    def _Xy2predicttions(self, X, y):
        if self.task == 'regression' or self.method == 'hard':
            np = self.np
            est_predictions = np.zeros((len(y), len(self.estimators)), dtype=np.float64)
        else:
            est_predictions = None

        iterators = self._cross_validator()
        for fold, (train, test) in enumerate(iterators.split(X, y)):
            for n, estimator in enumerate(self.estimators):
                X_train = X.iloc[train]
                y_train = y.iloc[train]
                X_test = X.iloc[test]
                estimator.fit(X_train, y_train)
                if self.classes_ is None and hasattr(estimator, 'classes_'):
                    self.classes_ = estimator.classes_
                pred = self._estimator_predict(estimator, X_test)
                if est_predictions is None:
                    np = self.np
                    est_predictions = np.zeros((len(y), len(self.estimators), pred.shape[1]), dtype=np.float64)
                est_predictions[test, n] = pred
        return est_predictions

    def _X2predictions(self, X):
        np = self.np
        if self.task == 'regression' or self.method == 'hard':
            est_predictions = np.zeros((len(X), len(self.estimators)), dtype=np.float64)
        else:
            est_predictions = np.zeros((len(X), len(self.estimators), len(self.classes_)), dtype=np.float64)

        for n, estimator in enumerate(self.estimators):
            if estimator is not None:
                pred = self._estimator_predict(estimator, X)
                if self.task == 'regression' and len(pred.shape) > 1:
                    assert pred.shape[1] == 1
                    pred = pred.reshape(pred.shape[0])
                est_predictions[:, n] = pred
        return est_predictions

    def _indices2predict(self, indices):
        assert self.classes_ is not None

        from .. import get_tool_box
        tb = get_tool_box(indices)
        return tb.take_array(self.classes_, indices, axis=0)

    def predict(self, X):
        est_predictions = self._X2predictions(X)
        pred = self.predictions2predict(est_predictions)
        if self.task != 'regression' and self.classes_ is not None:
            # np = self.np
            # pred = np.take(np.array(self.classes_), pred, axis=0)
            pred = self._indices2predict(pred)
        return pred

    def predict_proba(self, X):
        est_predictions = self._X2predictions(X)
        return self.predictions2predict_proba(est_predictions)

    def fit_predictions(self, predictions, y_true):
        raise NotImplementedError()

    def predictions2predict_proba(self, predictions):
        raise NotImplementedError()

    def predictions2predict(self, predictions):
        raise NotImplementedError()

    def save(self, model_path):
        if not model_path.endswith(fs.sep):
            model_path = model_path + fs.sep
        if not fs.exists(model_path):
            fs.mkdirs(model_path, exist_ok=True)

        stub = copy.copy(self)
        estimators = self.estimators
        if estimators is not None:
            stub.estimators = [None for _ in estimators]  # keep size

        if estimators is not None:
            for i, est in enumerate(estimators):
                est_pkl = f'{model_path}{i}.pkl'
                est_model = f'{model_path}{i}.model'
                for t in [est_pkl, est_model]:
                    if fs.exists(t):
                        fs.rm(t)

                if est is None:
                    continue
                with fs.open(est_pkl, 'wb') as f:
                    pickle.dump(est, f, protocol=pickle.HIGHEST_PROTOCOL)

                if hasattr(est, 'save') and hasattr(est, 'load'):
                    est.save(est_model)

        with fs.open(f'{model_path}ensemble.pkl', 'wb') as f:
            pickle.dump(stub, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(model_path):
        if not model_path.endswith(fs.sep):
            model_path = model_path + fs.sep

        with fs.open(f'{model_path}ensemble.pkl', 'rb') as f:
            stub = pickle.load(f)

        if stub.estimators is not None:
            for i in range(len(stub.estimators)):
                if fs.exists(f'{model_path}{i}.pkl'):
                    with fs.open(f'{model_path}{i}.pkl', 'rb') as f:
                        est = pickle.load(f)
                    if fs.exists(f'{model_path}{i}.model') and hasattr(est, 'load'):
                        est = est.load(f'{model_path}{i}.model')
                    stub.estimators[i] = est

        return stub
