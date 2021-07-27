# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import copy
import sys
import time

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm.sklearn import LGBMClassifier
from sklearn import model_selection as sksel
from sklearn.metrics import roc_auc_score, matthews_corrcoef, make_scorer

from hypernets.tabular import dask_ex as dex
from hypernets.tabular.column_selector import column_object_category_bool, column_number_exclude_timedelta
from hypernets.utils import logging
from .cfg import TabularCfg as cfg
from .general import general_preprocessor, general_estimator

logger = logging.getLogger(__name__)

is_os_windows = sys.platform.find('win') >= 0
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)


class FeatureSelectionCallback():
    def on_round_start(self, round_no, features, ):
        pass

    def on_round_end(self, round_no, auc, features, remove_features, elapsed):
        pass

    def on_remove_shift_variable(self, shift_score, remove_features):
        pass

    def on_task_break(self, round_no, auc, features):
        pass

    def on_task_finished(self, round_no, auc, features):
        pass


def feature_selection(X_train, X_test,
                      remove_shift_variable=True,
                      variable_shift_threshold=0.7,
                      variable_shift_scorer=None,
                      auc_threshold=0.55,
                      min_features=10, remove_size=0.1,
                      preprocessor=None,
                      estimator=None, sample_balance=True, max_test_samples=None, cv=5, random_state=9527,
                      copy_data=False,
                      callbacks=None):
    logger.info('Feature selection to try to eliminate the concept drift.')
    if copy_data:
        X_train = X_train.copy() if isinstance(X_train, dd.DataFrame) else copy.deepcopy(X_train)
        X_test = X_test.copy() if isinstance(X_test, dd.DataFrame) else copy.deepcopy(X_test)

    scores = None
    if remove_shift_variable:
        scores = covariate_shift_score(X_train, X_test, scorer=variable_shift_scorer)
        remain_features = []
        remove_features = []
        for col, score in scores.items():
            if score <= variable_shift_threshold:
                remain_features.append(col)
            else:
                remove_features.append(col)
                logger.info(f'Remove shift variables:{col},  score:{score}')
        if len(remain_features) < X_train.shape[1]:
            X_train = X_train[remain_features]
            X_test = X_test[remain_features]
        if callbacks is not None:
            for callback in callbacks:
                callback.on_remove_shift_variable(scores, remove_features)
    round = 1
    history = []
    while True:
        start_time = time.time()
        if callbacks is not None:
            for callback in callbacks:
                callback.on_round_start(round_no=round, features=X_train.columns.to_list())
        logger.info(f'\nRound: {round}\n')
        detector = DriftDetector(preprocessor, estimator, random_state)
        detector.fit(X_train, X_test, sample_balance=sample_balance, max_test_samples=max_test_samples, cv=cv)
        logger.info(f'AUC:{detector.auc_}, Features:{detector.feature_names_}')
        elapsed = time.time() - start_time
        history.append({'auc': detector.auc_,
                        'n_features': len(detector.feature_names_),
                        'removed_features': [],
                        'feature_names': detector.feature_names_,
                        'feature_importances': detector.feature_importances_,
                        'elapsed': elapsed
                        })

        if detector.auc_ <= auc_threshold:
            logger.info(
                f'AUC:{detector.auc_} has dropped below the threshold:{auc_threshold}, feature selection is over.')
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_task_finished(round_no=round, auc=detector.auc_, features=detector.feature_names_)
            return detector.feature_names_, history, scores

        indices = np.argsort(detector.feature_importances_)
        if indices.shape[0] <= min_features:
            logger.info(f'The number of remaining features is insufficient to continue remove features. '
                        f'AUC:{detector.auc_} '
                        f'Remaining features:{detector.feature_names_}')
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_task_break(round_no=round, auc=detector.auc_, features=detector.feature_names_)
            return detector.feature_names_, history, scores

        removes = int(indices.shape[0] * remove_size)
        if removes <= 0:
            logger.info(f'The number of remaining features is insufficient to continue remove features. '
                        f'AUC:{detector.auc_} '
                        f'Remaining features:({len(detector.feature_names_)}) / {detector.feature_names_}')
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_task_break(round_no=round, auc=detector.auc_, features=detector.feature_names_)
            return detector.feature_names_, history, scores

        if (indices.shape[0] - removes) < min_features:
            removes = indices.shape[0] - min_features

        remain_features = list(np.array(detector.feature_names_)[indices[:-removes]])
        remove_features = list(set(detector.feature_names_) - set(remain_features))
        history[-1]['removed_features'] = remove_features

        logger.info(f'Removed features: {remove_features}')
        X_train = X_train[remain_features]
        X_test = X_test[remain_features]
        if callbacks is not None:
            for callback in callbacks:
                callback.on_round_end(round_no=round, auc=detector.auc_, features=detector.feature_names_,
                                      remove_features=remove_features, elapsed=elapsed)
        round += 1


class DriftDetector():
    def __init__(self, preprocessor=None, estimator=None, random_state=9527):
        self.preprocessor = preprocessor
        self.estimator_ = estimator
        self.random_state = random_state
        self.auc_ = None
        self.feature_names_ = None
        self.feature_importances_ = None
        self.fitted = False

    def fit(self, X_train, X_test, sample_balance=True, max_test_samples=None, cv=5):
        logger.info('Fit data for concept drift detection')
        assert X_train.shape[1] == X_test.shape[1], 'The number of columns in X_train and X_test must be the same.'
        assert len(set(X_train.columns.to_list()) - set(
            X_test.columns.to_list())) == 0, 'The name of columns in X_train and X_test must be the same.'

        if dex.exist_dask_dataframe(X_train, X_test):
            import dask_ml.model_selection as dsel
            train_shape, test_shape = dask.compute(X_train.shape, X_test.shape)
            iterators = dsel.KFold(n_splits=cv, shuffle=True, random_state=1001)
        else:
            train_shape, test_shape = X_train.shape, X_test.shape
            iterators = sksel.StratifiedKFold(n_splits=cv, shuffle=True, random_state=1001)

        if max_test_samples is not None and max_test_samples < test_shape[0]:
            X_test, _ = dex.train_test_split(X_test, train_size=max_test_samples, random_state=self.random_state)
            test_shape = (max_test_samples, test_shape[0])
        if sample_balance:
            if test_shape[0] > train_shape[0]:
                X_test, _ = dex.train_test_split(X_test, train_size=train_shape[0], random_state=self.random_state)
                test_shape = (train_shape[0], test_shape[1])
            elif test_shape[0] < train_shape[0]:
                X_train, _ = dex.train_test_split(X_train, train_size=test_shape[0], random_state=self.random_state)
                train_shape = (test_shape[0], train_shape[1])

        target_col = '__drift_detection_target__'

        if hasattr(X_train, 'insert'):
            X_train.insert(0, target_col, 0)
        else:
            X_train[target_col] = 0

        if hasattr(X_test, 'insert'):
            X_test.insert(0, target_col, 1)
        else:
            X_test[target_col] = 1

        X_merge = dex.concat_df([X_train, X_test], repartition=True)
        y = X_merge.pop(target_col)

        logger.info('Preprocessing...')

        if self.preprocessor is None:
            self.preprocessor = general_preprocessor(X_merge)
        X_merge = self.preprocessor.fit_transform(X_merge)

        self.feature_names_ = X_merge.columns.to_list()
        self.feature_importances_ = []
        auc_all = []
        importances = []
        estimators = []

        if dex.is_dask_dataframe(X_merge):
            X_values, y_values = X_merge.to_dask_array(lengths=True), y.to_dask_array(lengths=True)
        else:
            # X_values, y_values = X_merge.values, y.values
            X_values, y_values = X_merge, y

        for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X_values, y_values)):
            logger.info(f'Fold:{n_fold + 1}')
            if dex.is_dask_dataframe(X_merge):
                x_train_fold, y_train_fold = X_values[train_idx], y_values[train_idx]
                x_val_fold, y_val_fold = X_values[valid_idx], y_values[valid_idx]
                x_train_fold = dex.array_to_df(x_train_fold, meta=X_merge)
                x_val_fold = dex.array_to_df(x_val_fold, meta=X_merge)
            else:
                x_train_fold, y_train_fold = X_values.iloc[train_idx], y_values.iloc[train_idx]
                x_val_fold, y_val_fold = X_values.iloc[valid_idx], y_values.iloc[valid_idx]

            estimator = general_estimator(X_merge, self.estimator_)
            kwargs = {}
            # if isinstance(estimator, dask_xgboost.XGBClassifier):
            #     kwargs['eval_set'] = [(x_val_fold.compute(), y_val_fold.compute())]
            #     kwargs['early_stopping_rounds'] = 10
            if isinstance(estimator, LGBMClassifier):
                kwargs['eval_set'] = (x_val_fold, y_val_fold)
                kwargs['early_stopping_rounds'] = 10
                kwargs['verbose'] = 0
            estimator.fit(x_train_fold, y_train_fold, **kwargs)
            proba = estimator.predict_proba(x_val_fold)[:, 1]
            if dex.is_dask_dataframe(X_merge):
                y_val_fold, proba = dex.compute(y_val_fold, proba, traverse=False)
            auc = roc_auc_score(y_val_fold, proba)
            logger.info(f'auc: {auc}')

            auc_all.append(auc)
            estimators.append(estimator)
            importances.append(estimator.feature_importances_)

        self.estimator_ = estimators
        self.auc_ = np.mean(auc_all)
        self.feature_importances_ = np.mean(importances, axis=0)
        self.fitted = True
        X_test.pop(target_col)
        X_train.pop(target_col)
        return self

    def predict_proba(self, X):
        assert self.fitted, 'Please fit it first.'

        X = X.copy() if dex.is_dask_dataframe(X) else copy.deepcopy(X)
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
        if dex.exist_dask_object(X, y):
            return self.train_test_split_by_dask(X, y, test_size=test_size, remain_for_train=remain_for_train)

        assert remain_for_train < 1.0 and remain_for_train >= 0, '`remain_for_train` must be < 1.0 and >= 0.'
        if isinstance(test_size, float):
            assert test_size < 1.0 and test_size > 0, '`test_size` must be < 1.0 and > 0.'
            test_size = int(X.shape[0] * test_size)
        assert isinstance(test_size, int), '`test_size` can only be int or float'
        split_size = int(test_size + test_size * remain_for_train)
        assert split_size < X.shape[0], \
            'test_size+test_size*remain_for_train must be less than the number of samples in X.'

        proba = self.predict_proba(X)
        sorted_indices = np.argsort(proba)
        target = '__train_test_split_y__'
        X.insert(0, target, y)

        if remain_for_train == 0:
            X_train = X.iloc[sorted_indices[:-test_size]]
            X_test = X.iloc[sorted_indices[-test_size:]]
            y_train = X_train.pop(target)
            y_test = X_test.pop(target)
            return X_train, X_test, y_train, y_test
        else:
            X_train_1 = X.iloc[sorted_indices[:-split_size]]
            X_mixed = X.iloc[sorted_indices[-split_size:]]
            X_train_2, X_test = dex.train_test_split(X_mixed, test_size=test_size, shuffle=True,
                                                     random_state=self.random_state)
            X_train = pd.concat([X_train_1, X_train_2], axis=0)
            y_train = X_train.pop(target)
            y_test = X_test.pop(target)
            return X_train, X_test, y_train, y_test

    def train_test_split_by_dask(self, X, y, test_size=0.25, remain_for_train=0.3):
        x_shape = dex.compute(X.shape)[0]
        assert remain_for_train < 1.0 and remain_for_train >= 0, '`remain_for_train` must be < 1.0 and >= 0.'
        if isinstance(test_size, float):
            assert test_size < 1.0 and test_size > 0, '`test_size` must be < 1.0 and > 0.'
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


def covariate_shift_score(X_train, X_test, scorer=None, cv=None, copy_data=True):
    assert all(isinstance(x, (pd.DataFrame, dd.DataFrame)) for x in (X_train, X_test)), \
        'X_train and X_test must be a pandas or dask DataFrame.'

    assert len(set(X_train.columns.to_list()) - set(
        X_test.columns.to_list())) == 0, 'The columns in X_train and X_test must be the same.'
    if scorer is None:
        scorer = roc_auc_scorer
    if copy_data:
        train = X_train.copy() if dex.is_dask_dataframe(X_train) else copy.deepcopy(X_train)
        test = X_test.copy() if dex.is_dask_dataframe(X_test) else copy.deepcopy(X_test)
    else:
        train = X_train
        test = X_test

    # Set target value
    target_col = '__hypernets_csd__target__'
    if hasattr(train, 'insert'):
        train.insert(0, target_col, 0)
    else:
        train[target_col] = 0
    if hasattr(test, 'insert'):
        test.insert(0, target_col, 1)
    else:
        test[target_col] = 1

    X_merge = dex.concat_df([train, test], axis=0)
    y = X_merge.pop(target_col)
    if dex.is_dask_dataframe(X_merge):
        y = y.compute()

    logger.info('Preprocessing...')
    # Preprocess data: imputing and scaling
    preprocessor = general_preprocessor(X_merge)
    X_merge = preprocessor.fit_transform(X_merge)
    if dex.is_dask_dataframe(X_merge):
        X_merge = X_merge.persist()

    # Calculate the shift score for each column separately.
    scores = {}
    logger.info('Scoring...')
    if dex.is_dask_dataframe(X_merge) or cfg.joblib_njobs in {0, 1}:
        for c in X_merge.columns:
            x = X_merge[[c]]
            if dex.is_dask_dataframe(X_merge):
                x = x.compute()
            score = _shift_score(x, y, scorer, cv)
            logger.info(f'column:{c}, score:{score}')
            scores[c] = score
    else:
        col_parts = [X_merge[[c]] for c in X_merge.columns]
        options = dict(backend='multiprocessing') if is_os_windows else dict(prefer='processes')
        pss = Parallel(n_jobs=cfg.joblib_njobs, **options)(delayed(_shift_score)(x, y, scorer, cv) for x in col_parts)
        scores = {k: v for k, v in zip(X_merge.columns.to_list(), pss)}
        logger.info(f'scores: {scores}')

    return scores


def _shift_score(x, y, scorer, cv):
    model = general_estimator(x)
    if cv:
        score_ = sksel.cross_val_score(model, X=x, y=y, verbose=0, scoring=scorer, cv=cv)
        score = np.mean(score_)
    else:
        mixed_x_train, mixed_x_test, mixed_y_train, mixed_y_test = \
            sksel.train_test_split(x, y, test_size=0.3, random_state=9527, stratify=y)
        model.fit(mixed_x_train, mixed_y_train,
                  eval_set=(mixed_x_test, mixed_y_test),
                  early_stopping_rounds=20,
                  verbose=False)
        score = scorer(model, mixed_x_test, mixed_y_test)

    return score
