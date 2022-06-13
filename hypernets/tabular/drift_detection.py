# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import copy
import inspect
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn import model_selection as sksel
from sklearn.metrics import roc_auc_score, matthews_corrcoef, make_scorer

from hypernets.core import randint
from hypernets.utils import logging, const
from .cfg import TabularCfg as cfg

logger = logging.getLogger(__name__)

roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)


class FeatureSelectionCallback:
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


def _shift_score(X, y, scorer, cv, log_level):
    from . import get_tool_box

    if log_level is not None:
        import warnings

        warnings.filterwarnings('ignore')
        logging.set_level(log_level)

    tb = get_tool_box(X, y)
    model = tb.general_estimator(X, y, task=const.TASK_BINARY)
    if cv:
        if scorer is None:
            scorer = roc_auc_scorer
        score_ = sksel.cross_val_score(model, X=X, y=y, verbose=0, scoring=scorer, cv=cv)
        score = np.mean(score_)
    else:
        X_train, X_test, y_train, y_test = \
            tb.train_test_split(X, y, test_size=0.3, random_state=9527, stratify=y)
        model.fit(X_train, y_train, **_get_fit_kwargs(model, X_eval=X_test, y_eval=y_test))
        if scorer:
            score = scorer(model, X_test, y_test)
        else:
            y_proba = model.predict_proba(X_test)
            score = tb.metrics.calc_score(y_test, y_proba, y_proba, metrics=['auc'])
            assert 'auc' in score.keys()
            score = score['auc']

    return score


def _get_fit_kwargs(estimator, *, X_eval, y_eval, early_stopping_rounds=20, verbose=False):
    kwargs = {}
    fit_params = inspect.signature(estimator.fit).parameters.keys()
    if 'eval_set' in fit_params and X_eval is not None and y_eval is not None:
        kwargs['eval_set'] = [(X_eval, y_eval)]
    if 'early_stopping_rounds' in fit_params and early_stopping_rounds is not None:
        if type(estimator).__name__.find('DaskLGBM') < 0:  # DaskLGBMXxx dose not support early_stopping_rounds now
            kwargs['early_stopping_rounds'] = early_stopping_rounds
    if 'verbose' in fit_params:
        kwargs['verbose'] = verbose
    return kwargs


def _detect_options(estimator, method, **kwargs):
    r = {}
    params = inspect.signature(getattr(estimator, method)).parameters.keys()

    for k, v in kwargs.items():
        if k in params:
            r[k] = v

    return r


class DriftDetector:
    def __init__(self, preprocessor=None, estimator=None, random_state=None):
        self.preprocessor = preprocessor
        self.estimator_ = estimator
        self.random_state = random_state if random_state is not None else randint()
        self.auc_ = None
        self.feature_names_ = None
        self.feature_importances_ = None
        self.fitted = False

    def fit(self, X_train, X_test, sample_balance=True, max_test_samples=None, cv=5):
        logger.info('Fit data for concept drift detection')
        assert X_train.shape[1] == X_test.shape[1], 'The number of columns in X_train and X_test must be the same.'
        assert len(set(X_train.columns.to_list()) - set(X_test.columns.to_list())) == 0, \
            'The name of columns in X_train and X_test must be the same.'

        from . import get_tool_box

        train_size, test_size = len(X_train), len(X_test)
        tb = get_tool_box(X_train, X_test)

        if max_test_samples is not None and max_test_samples < test_size:
            X_test, _ = tb.train_test_split(X_test, train_size=max_test_samples, random_state=self.random_state)
            test_size = len(X_test)

        if sample_balance:
            if test_size > train_size:
                X_test, _ = tb.train_test_split(X_test, train_size=train_size, random_state=self.random_state)
            elif test_size < train_size:
                X_train, _ = tb.train_test_split(X_train, train_size=test_size, random_state=self.random_state)

        logger.info('Merge train and test data...')
        X_merged, y = self._train_test_merge(X_train, X_test)

        logger.info('Preprocessing...')
        if self.preprocessor is None:
            self.preprocessor = tb.general_preprocessor(X_merged)
        X_merged = self.preprocessor.fit_transform(X_merged, y)

        logger.info('Fitting and scoring...')
        estimators, auc_all, importances = self._fit_and_score(X_merged, y, cv=cv)

        self.feature_names_ = X_merged.columns.to_list()
        self.estimator_ = estimators
        self.auc_ = np.mean(auc_all)
        self.feature_importances_ = np.mean(importances, axis=0)
        self.fitted = True

        return self

    def _fit_and_score(self, X_merged, y_merged, *, cv):
        from . import get_tool_box

        auc_all = []
        importances = []
        estimators = []

        tb = get_tool_box(X_merged, y_merged)
        sel = tb.select_1d
        iterators = tb.statified_kfold(n_splits=cv, shuffle=True, random_state=self.random_state)
        for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X_merged, y_merged)):
            logger.info(f'Fold:{n_fold + 1}')
            x_train_fold, y_train_fold = sel(X_merged, train_idx), sel(y_merged, train_idx)
            x_val_fold, y_val_fold = sel(X_merged, valid_idx), sel(y_merged, valid_idx)

            estimator = tb.general_estimator(X_merged, y_merged, self.estimator_, task=const.TASK_BINARY)
            # kwargs = {}
            # estimator_type = type(estimator).__name__
            # if estimator_type.find('LGBMClassifier') >= 0:
            #     kwargs['eval_set'] = [(x_val_fold, y_val_fold)]
            #     kwargs['early_stopping_rounds'] = 10
            #     kwargs['verbose'] = 0
            kwargs = _get_fit_kwargs(estimator, X_eval=x_val_fold, y_eval=y_val_fold)
            estimator.fit(x_train_fold, y_train_fold, **kwargs)
            kwargs = _detect_options(estimator, 'predict_proba', to_local=True, verbose=False)
            proba = estimator.predict_proba(x_val_fold, **kwargs)[:, 1]
            y_val_fold, proba = tb.to_local(y_val_fold, proba)
            auc = roc_auc_score(y_val_fold, proba)
            logger.info(f'auc: {auc}')

            auc_all.append(auc)
            estimators.append(estimator)
            importances.append(estimator.feature_importances_)

        return estimators, auc_all, importances

    def predict_proba(self, X):
        assert self.fitted, 'Please fit it first.'

        from . import get_tool_box
        tb = get_tool_box(X)

        X = self._copy_data(X)
        cat_cols = tb.column_selector.column_object_category_bool(X)
        num_cols = tb.column_selector.column_number_exclude_timedelta(X)
        # X.loc[:, cat_cols + num_cols] = self.preprocessor.transform(X)
        Xt = self.preprocessor.transform(X)
        diff_cols = set(X.columns.tolist()) - set(cat_cols + num_cols)
        if diff_cols:
            # X.loc[:, cat_cols + num_cols] = Xt
            X = tb.concat_df([X[diff_cols], Xt[cat_cols + num_cols]], axis=1)
        else:
            X = Xt

        oof_proba = []
        for i, estimator in enumerate(self.estimator_):
            proba = estimator.predict_proba(X)[:, 1]
            oof_proba.append(proba)
        # proba = np.mean(oof_proba, axis=0)
        proba = tb.mean_oof(oof_proba)

        return proba

    def train_test_split(self, X, y, test_size=0.25, remain_for_train=0.3):
        assert 0 <= remain_for_train < 1.0, '`remain_for_train` must be < 1.0 and >= 0.'
        if isinstance(test_size, float):
            assert 0 < test_size < 1.0, '`test_size` must be < 1.0 and > 0.'
            test_size = int(len(X) * test_size)
        assert isinstance(test_size, int), '`test_size` can only be int or float'
        split_size = int(test_size + test_size * remain_for_train)
        assert split_size < len(X), \
            'test_size+test_size*remain_for_train must be less than the number of samples in X.'

        from . import get_tool_box
        tb = get_tool_box(X, y)
        sel = tb.select_1d

        proba = self.predict_proba(X)
        proba, = tb.to_local(proba)
        sorted_indices = np.argsort(proba)

        target_col = '__train_test_split_y__'
        if hasattr(X, 'insert'):
            X.insert(0, target_col, y)
        else:
            X[target_col] = y

        if remain_for_train == 0:
            X_train = sel(X, sorted_indices[:-test_size])
            X_test = sel(X, sorted_indices[-test_size:])
        else:
            X_train_1 = sel(X, sorted_indices[:-split_size])
            X_mixed = sel(X, sorted_indices[-split_size:])
            X_train_2, X_test = tb.train_test_split(
                X_mixed, test_size=test_size, shuffle=True, random_state=self.random_state)
            X_train = tb.concat_df([X_train_1, X_train_2], axis=0)

        y_train = X_train.pop(target_col)
        y_test = X_test.pop(target_col)
        X.pop(target_col)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _copy_data(X):
        return copy.deepcopy(X)

    def _train_test_merge(self, X_train, X_test, shuffle=True):
        from . import get_tool_box

        target_col = '__hypernets_tmp__target__'
        if hasattr(X_train, 'insert'):
            X_train.insert(0, target_col, 0)
        else:
            X_train[target_col] = 0
        if hasattr(X_test, 'insert'):
            X_test.insert(0, target_col, 1)
        else:
            X_test[target_col] = 1

        tb = get_tool_box(X_train, X_test)
        X_merge = tb.concat_df([X_train, X_test], axis=0, repartition=shuffle, random_state=self.random_state)
        y = X_merge.pop(target_col)

        X_train.pop(target_col)
        X_test.pop(target_col)

        return X_merge, y


class FeatureSelectorWithDriftDetection:
    parallelizable = True

    def __init__(self, remove_shift_variable=True, variable_shift_threshold=0.7, variable_shift_scorer=None,
                 auc_threshold=0.55, min_features=10, remove_size=0.1,
                 sample_balance=True, max_test_samples=None, cv=5, random_state=None,
                 callbacks=None):
        self.remove_shift_variable = remove_shift_variable
        self.variable_shift_threshold = variable_shift_threshold
        self.variable_shift_scorer = variable_shift_scorer
        self.auc_threshold = auc_threshold
        self.min_features = min_features
        self.remove_size = remove_size
        self.sample_balance = sample_balance
        self.max_test_samples = max_test_samples
        self.cv = cv
        self.random_state = random_state if random_state is not None else randint()
        self.callbacks = callbacks

    def select(self, X_train, X_test, *, preprocessor=None, estimator=None, copy_data=False):
        logger.info('Feature selection to try to eliminate the concept drift.')

        if copy_data:
            detector = self.get_detector(preprocessor, estimator, self.random_state)
            X_train = detector._copy_data(X_train)
            X_test = detector._copy_data(X_test)

        scores = None
        if self.remove_shift_variable:
            scores = self._covariate_shift_score(X_train, X_test, scorer=self.variable_shift_scorer)
            remain_features = []
            remove_features = []
            for col, score in scores.items():
                if score <= self.variable_shift_threshold:
                    remain_features.append(col)
                else:
                    remove_features.append(col)
                    logger.info(f'Remove shift variables:{col},  score:{score}')
            if len(remain_features) < X_train.shape[1]:
                X_train = X_train[remain_features]
                X_test = X_test[remain_features]
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_remove_shift_variable(scores, remove_features)

        round = 1
        history = []
        while True:
            start_time = time.time()
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_round_start(round_no=round, features=X_train.columns.to_list())
            logger.info(f'\nRound: {round}\n')
            detector = self.get_detector(preprocessor, estimator, self.random_state)
            detector.fit(X_train, X_test,
                         sample_balance=self.sample_balance,
                         max_test_samples=self.max_test_samples,
                         cv=self.cv)
            logger.info(f'AUC:{detector.auc_}, Features:{detector.feature_names_}')
            elapsed = time.time() - start_time
            history.append({'auc': detector.auc_,
                            'n_features': len(detector.feature_names_),
                            'removed_features': [],
                            'feature_names': detector.feature_names_,
                            'feature_importances': detector.feature_importances_,
                            'elapsed': elapsed
                            })

            if detector.auc_ <= self.auc_threshold:
                logger.info(
                    f'AUC:{detector.auc_} has dropped below the threshold:{self.auc_threshold}, feature selection is over.')
                if self.callbacks is not None:
                    for callback in self.callbacks:
                        callback.on_task_finished(round_no=round, auc=detector.auc_, features=detector.feature_names_)
                return detector.feature_names_, history, scores

            indices = np.argsort(detector.feature_importances_)
            if indices.shape[0] <= self.min_features:
                logger.info(f'The number of remaining features is insufficient to continue remove features. '
                            f'AUC:{detector.auc_} '
                            f'Remaining features:{detector.feature_names_}')
                if self.callbacks is not None:
                    for callback in self.callbacks:
                        callback.on_task_break(round_no=round, auc=detector.auc_, features=detector.feature_names_)
                return detector.feature_names_, history, scores

            removes = int(indices.shape[0] * self.remove_size)
            if removes <= 0:
                logger.info(f'The number of remaining features is insufficient to continue remove features. '
                            f'AUC:{detector.auc_} '
                            f'Remaining features:({len(detector.feature_names_)}) / {detector.feature_names_}')
                if self.callbacks is not None:
                    for callback in self.callbacks:
                        callback.on_task_break(round_no=round, auc=detector.auc_, features=detector.feature_names_)
                return detector.feature_names_, history, scores

            if (indices.shape[0] - removes) < self.min_features:
                removes = indices.shape[0] - self.min_features

            remain_features = list(np.array(detector.feature_names_)[indices[:-removes]])
            remove_features = list(set(detector.feature_names_) - set(remain_features))
            history[-1]['removed_features'] = remove_features

            logger.info(f'Removed features: {remove_features}')
            X_train = X_train[remain_features]
            X_test = X_test[remain_features]
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_round_end(round_no=round, auc=detector.auc_, features=detector.feature_names_,
                                          remove_features=remove_features, elapsed=elapsed)
            round += 1

    def _covariate_shift_score(self, X_train, X_test, *,
                               preprocessor=None, estimator=None, scorer=None, cv=None,
                               shuffle=True, copy_data=True):
        from . import get_tool_box

        # assert all(isinstance(x, (pd.DataFrame, dd.DataFrame)) for x in (X_train, X_test)), \
        #     'X_train and X_test must be a pandas or dask DataFrame.'
        assert set(X_train.columns.to_list()) == set(X_test.columns.to_list()), \
            'The columns in X_train and X_test must be the same.'

        detector = self.get_detector(preprocessor, estimator, self.random_state)

        if copy_data:
            X_train = detector._copy_data(X_train)
            X_test = detector._copy_data(X_test)

        shift_variable_sample_limit = cfg.shift_variable_sample_limit
        if len(X_train) > shift_variable_sample_limit:
            logger.info(f'sample X_train to {shift_variable_sample_limit}')
            X_train = get_tool_box(X_train).select_1d(X_train, np.arange(shift_variable_sample_limit))
        if len(X_test) > shift_variable_sample_limit:
            logger.info(f'sample X_test to {shift_variable_sample_limit}')
            X_test = get_tool_box(X_test).select_1d(X_test, np.arange(shift_variable_sample_limit))

        # Set target value
        logger.info('Set target value...')
        X_merged, y = detector._train_test_merge(X_train, X_test, shuffle=shuffle)

        logger.info('Preprocessing...')
        # Preprocess data: imputing and scaling
        if preprocessor is None:
            preprocessor = get_tool_box(X_merged).general_preprocessor(X_merged)
        X_merged = preprocessor.fit_transform(X_merged)

        # Calculate the shift score for each column separately.
        logger.info('Scoring...')
        scores = self._score_features(X_merged, y, scorer, cv)

        return scores

    def _score_features(self, X_merged, y, scorer, cv):
        if not self.parallelizable or cfg.joblib_njobs in {0, 1}:
            scores = {}
            for c in X_merged.columns:
                x = X_merged[[c]]
                score = _shift_score(x, y, scorer, cv, None)
                logger.info(f'column:{c}, score:{score}')
                scores[c] = score
        else:
            log_level = logging.get_level()
            col_parts = [X_merged[[c]] for c in X_merged.columns]
            pss = Parallel(n_jobs=cfg.joblib_njobs, **cfg.joblib_options)(
                delayed(_shift_score)(x, y, scorer, cv, log_level) for x in col_parts
            )
            scores = {k: v for k, v in zip(X_merged.columns.to_list(), pss)}
            logger.info(f'scores: {scores}')

        return scores

    @staticmethod
    def get_detector(preprocessor=None, estimator=None, random_state=None):
        return DriftDetector(preprocessor=preprocessor, estimator=estimator, random_state=random_state)
