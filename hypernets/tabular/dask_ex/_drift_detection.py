# -*- coding:utf-8 -*-
"""

"""
import dask.array as da
import numpy as np
from sklearn.metrics import roc_auc_score

from hypernets.tabular import dask_ex as dex
from hypernets.tabular.column_selector import column_object_category_bool, column_number_exclude_timedelta
from hypernets.utils import logging, const
from .. import get_tool_box
from ..drift_detection import FeatureSelectorWithDriftDetection, DriftDetector, _shift_score

logger = logging.getLogger(__name__)


class DaskFeatureSelectionWithDriftDetector(FeatureSelectorWithDriftDetection):
    @staticmethod
    def get_detector(preprocessor=None, estimator=None, random_state=9527):
        return DaskDriftDetector(preprocessor=preprocessor, estimator=estimator, random_state=random_state)

    @staticmethod
    def _score_features(X_merged, y, scorer, cv):
        if dex.is_dask_object(y):
            y = y.compute()

        if not dex.is_dask_object(X_merged):
            return super()._score_features(X_merged, y, scorer, cv)

        scores = {}
        for c in X_merged.columns:
            x = X_merged[[c]].compute()
            score = _shift_score(x, y, scorer, cv)
            logger.info(f'column:{c}, score:{score}')
            scores[c] = score

        return scores


class DaskDriftDetector(DriftDetector):
    @staticmethod
    def _copy_data(X):
        return X.copy()

    @staticmethod
    def _train_test_merge(X_train, X_test):
        target_col = '__hypernets_tmp__target__'
        X_train[target_col] = 0
        X_test[target_col] = 1

        X_merge = dex.concat_df([X_train, X_test], axis=0, repartition=True)
        y = X_merge.pop(target_col)

        return X_merge, y

    def _fit_and_score(self, X_merged, y_merged, *, cv):
        auc_all = []
        importances = []
        estimators = []
        tb = get_tool_box(X_merged)
        X_values, y_values = X_merged.to_dask_array(lengths=True), y_merged.to_dask_array(lengths=True)

        import dask_ml.model_selection as dsel
        iterators = dsel.KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X_values, y_values)):
            logger.info(f'Fold:{n_fold + 1}')
            x_train_fold, y_train_fold = X_values[train_idx], y_values[train_idx]
            x_val_fold, y_val_fold = X_values[valid_idx], y_values[valid_idx]
            x_train_fold = dex.array_to_df(x_train_fold, meta=X_merged)
            x_val_fold = dex.array_to_df(x_val_fold, meta=X_merged)

            estimator = tb.general_estimator(X_merged, y_merged, self.estimator_, task=const.TASK_BINARY)
            kwargs = {}
            if type(estimator).__name__.find('DaskLGBMClassifier') >= 0:
                kwargs['eval_set'] = (x_val_fold, y_val_fold)
                kwargs['early_stopping_rounds'] = 10
                kwargs['verbose'] = 0
            estimator.fit(x_train_fold, y_train_fold, **kwargs)
            proba = estimator.predict_proba(x_val_fold)[:, 1]
            y_val_fold, proba = dex.compute(y_val_fold, proba, traverse=False)
            auc = roc_auc_score(y_val_fold, proba)
            logger.info(f'auc: {auc}')

            auc_all.append(auc)
            estimators.append(estimator)
            importances.append(estimator.feature_importances_)

        return estimators, auc_all, importances

    def predict_proba(self, X):
        assert self.fitted, 'Please fit it first.'

        X = self._copy_data(X)
        cat_cols = column_object_category_bool(X)
        num_cols = column_number_exclude_timedelta(X)
        # X.loc[:, cat_cols + num_cols] = self.preprocessor.transform(X)
        Xt = self.preprocessor.transform(X)
        diff_cols = set(X.columns.tolist()) - set(cat_cols + num_cols)
        if diff_cols:
            # X.loc[:, cat_cols + num_cols] = Xt
            X = dex.concat_df([X[diff_cols], Xt[cat_cols + num_cols]], axis=1)
        else:
            X = Xt

        oof_proba = []
        for i, estimator in enumerate(self.estimator_):
            proba = estimator.predict_proba(X)[:, 1]
            oof_proba.append(proba)
        if dex.is_dask_dataframe(X):
            proba = da.mean(dex.hstack_array(oof_proba), axis=1)
        else:
            proba = np.mean(oof_proba, axis=0)
        return proba

    def train_test_split(self, X, y, test_size=0.25, remain_for_train=0.3):
        if not dex.exist_dask_object(X, y):
            return super().train_test_split(X, y, test_size=test_size, remain_for_train=remain_for_train)

        x_shape = dex.compute(X.shape)[0]
        assert 0 <= remain_for_train < 1.0, '`remain_for_train` must be < 1.0 and >= 0.'
        if isinstance(test_size, float):
            assert 0 < test_size < 1.0, '`test_size` must be < 1.0 and > 0.'
            test_size = int(x_shape[0] * test_size)
        assert isinstance(test_size, int), '`test_size` can only be int or float'
        split_size = int(test_size + test_size * remain_for_train)
        assert split_size < x_shape[0], \
            'test_size+test_size*remain_for_train must be less than the number of samples in X.'

        X = X.copy()
        proba = self.predict_proba(X)
        sorted_indices = np.argsort(proba.compute())
        target = '__train_test_split_y__'
        X[target] = y

        X_values = X.to_dask_array(lengths=True)
        if remain_for_train == 0:
            X_train = dex.array_to_df(X_values[sorted_indices[:-test_size]], meta=X)
            X_test = dex.array_to_df(X_values[sorted_indices[-test_size:]], meta=X)
            y_train = X_train.pop(target)
            y_test = X_test.pop(target)
            return X_train, X_test, y_train, y_test
        else:
            X_train_1 = dex.array_to_df(X_values[sorted_indices[:-split_size]], meta=X)
            X_mixed = dex.array_to_df(X_values[sorted_indices[-split_size:]], meta=X)
            X_train_2, X_test = dex.train_test_split(X_mixed, test_size=test_size, shuffle=True,
                                                     random_state=self.random_state)
            X_train = dex.concat_df([X_train_1, X_train_2], axis=0)
            y_train = X_train.pop(target)
            y_test = X_test.pop(target)
            return X_train, X_test, y_train, y_test
