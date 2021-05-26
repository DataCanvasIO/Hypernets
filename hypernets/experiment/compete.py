# -*- coding:utf-8 -*-
__author__ = 'yangjian'

"""

"""
import copy
import inspect
from collections import OrderedDict

import numpy as np
import pandas as pd
from IPython.display import display, display_markdown
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from hypernets.experiment import Experiment
from hypernets.tabular import dask_ex as dex
from hypernets.tabular import drift_detection as dd
from hypernets.tabular.cache import cache
from hypernets.tabular.data_cleaner import DataCleaner
from hypernets.tabular.ensemble import GreedyEnsemble, DaskGreedyEnsemble
from hypernets.tabular.feature_generators import FeatureGenerationTransformer
from hypernets.tabular.feature_importance import permutation_importance_batch, select_by_feature_importance
from hypernets.tabular.feature_selection import select_by_multicollinearity
from hypernets.tabular.general import general_estimator, general_preprocessor
from hypernets.tabular.lifelong_learning import select_valid_oof
from hypernets.tabular.pseudo_labeling import sample_by_pseudo_labeling
from hypernets.utils import logging, isnotebook, const

logger = logging.get_logger(__name__)

DEFAULT_EVAL_SIZE = 0.3

_is_notebook = isnotebook()


def _set_log_level(log_level):
    logging.set_level(log_level)

    # if log_level >= logging.ERROR:
    #     import logging as pylogging
    #     pylogging.basicConfig(level=log_level)


class ExperimentStep(BaseEstimator):
    def __init__(self, experiment, name):
        super(ExperimentStep, self).__init__()

        self.name = name
        self.experiment = experiment

        # fitted
        self.input_features_ = None
        self.status_ = None  # None(not fit) or True(fit succeed) or False(fit failed)

    def step_progress(self, *args, **kwargs):
        if self.experiment is not None:
            self.experiment.step_progress(*args, **kwargs)

    @property
    def task(self):
        return self.experiment.task if self.experiment is not None else None

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        self.input_features_ = X_train.columns.to_list()
        # self.status_ = True

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def transform(self, X, y=None, **kwargs):
        raise NotImplemented()
        # return X

    def is_transform_skipped(self):
        return False

    def get_fitted_params(self):
        return {'input_features': self.input_features_}

    # override this to remove 'experiment' from estimator __expr__
    @classmethod
    def _get_param_names(cls):
        params = super()._get_param_names()
        return filter(lambda x: x != 'experiment', params)

    def __getstate__(self):
        state = super().__getstate__()
        # Don't pickle experiment
        if 'experiment' in state.keys():
            state['experiment'] = None
        return state

    def _repr_df_(self):
        init_params = self.get_params()
        fitted_params = self.get_fitted_params()

        init_df = pd.Series(init_params, name='value').to_frame()
        init_df['kind'] = 'settings'

        fitted_df = pd.Series(fitted_params, name='value').to_frame()
        fitted_df['kind'] = 'fitted'

        df = pd.concat([init_df, fitted_df], axis=0)
        df['key'] = df.index
        df = df.set_index(['kind', 'key'])

        return df

    def _repr_html_(self):
        df = self._repr_df_()
        html = f'<h2>{self.name}</h2>{df._repr_html_()}'
        return html


class FeatureSelectStep(ExperimentStep):

    def __init__(self, experiment, name):
        super().__init__(experiment, name)

        # fitted
        self.selected_features_ = None

    def transform(self, X, y=None, **kwargs):
        if self.selected_features_ is not None:
            if logger.is_debug_enabled():
                msg = f'{self.name} transform from {len(X.columns.tolist())} to {len(self.selected_features_)} features'
                logger.debug(msg)
            X = X[self.selected_features_]
        return X

    def cache_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        if self.selected_features_ is not None:
            features = self.selected_features_
            X_train = X_train[features]
            if X_test is not None:
                X_test = X_test[features]
            if X_eval is not None:
                X_eval = X_eval[features]
            if logger.is_info_enabled():
                logger.info(f'{self.name} cache_transform: {len(X_train.columns)} columns kept.')
        else:
            if logger.is_info_enabled():
                logger.info(f'{self.name} cache_transform: {len(X_train.columns)} columns kept (do nothing).')

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def is_transform_skipped(self):
        return self.selected_features_ is None

    def get_fitted_params(self):
        if self.selected_features_ is None:
            unselected = None
        else:
            unselected = list(filter(lambda _: _ not in self.selected_features_, self.input_features_))

        return {**super().get_fitted_params(),
                'selected_features': self.selected_features_,
                'unselected_features': unselected}


class DataCleanStep(FeatureSelectStep):
    def __init__(self, experiment, name, data_cleaner_args=None,
                 cv=False, train_test_split_strategy=None, random_state=None):
        super().__init__(experiment, name)

        self.data_cleaner_args = data_cleaner_args if data_cleaner_args is not None else {}
        self.cv = cv
        self.train_test_split_strategy = train_test_split_strategy
        self.random_state = random_state

        # fitted
        self.data_cleaner_ = None
        self.detector_ = None
        self.data_shapes_ = None

    @cache(arg_keys='X_train,y_train,X_test,X_eval,y_eval',
           strategy='transform', transformer='cache_transform',
           attrs_to_restore='input_features_,selected_features_,data_cleaner_,detector_')
    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)

        # 1. Clean Data
        if self.cv and X_eval is not None and y_eval is not None:
            logger.info(f'{self.name} cv enabled, so concat train data and eval data')
            X_train = dex.concat_df([X_train, X_eval], axis=0)
            y_train = dex.concat_df([y_train, y_eval], axis=0)
            X_eval = None
            y_eval = None

        data_cleaner = DataCleaner(**self.data_cleaner_args)

        logger.info(f'{self.name} fit_transform with train data')
        X_train, y_train = data_cleaner.fit_transform(X_train, y_train)
        self.step_progress('fit_transform train set')

        if X_test is not None:
            logger.info(f'{self.name} transform test data')
            X_test = data_cleaner.transform(X_test)
            self.step_progress('transform X_test')

        if not self.cv:
            if X_eval is None or y_eval is None:
                eval_size = self.experiment.eval_size
                if self.train_test_split_strategy == 'adversarial_validation' and X_test is not None:
                    logger.debug('DriftDetector.train_test_split')
                    detector = dd.DriftDetector()
                    detector.fit(X_train, X_test)
                    self.detector_ = detector
                    X_train, X_eval, y_train, y_eval = \
                        detector.train_test_split(X_train, y_train, test_size=eval_size)
                else:
                    if self.task == const.TASK_REGRESSION or dex.is_dask_object(X_train):
                        X_train, X_eval, y_train, y_eval = \
                            dex.train_test_split(X_train, y_train, test_size=eval_size,
                                                 random_state=self.random_state)
                    else:
                        X_train, X_eval, y_train, y_eval = \
                            dex.train_test_split(X_train, y_train, test_size=eval_size,
                                                 random_state=self.random_state, stratify=y_train)
                if self.task != const.TASK_REGRESSION:
                    y_train_uniques = set(y_train.unique()) if hasattr(y_train, 'unique') else set(y_train)
                    y_eval_uniques = set(y_eval.unique()) if hasattr(y_eval, 'unique') else set(y_eval)
                    assert y_train_uniques == y_eval_uniques, \
                        'The classes of `y_train` and `y_eval` must be equal. Try to increase eval_size.'
                self.step_progress('split into train set and eval set')
            else:
                X_eval, y_eval = data_cleaner.transform(X_eval, y_eval)
                self.step_progress('transform eval set')

        selected_features = X_train.columns.to_list()
        data_shapes = {'X_train.shape': X_train.shape,
                       'y_train.shape': y_train.shape,
                       'X_eval.shape': None if X_eval is None else X_eval.shape,
                       'y_eval.shape': None if y_eval is None else y_eval.shape,
                       'X_test.shape': None if X_test is None else X_test.shape
                       }
        if dex.exist_dask_object(X_train, y_train, X_eval, y_eval, X_test):
            data_shapes = {k: dex.compute(v) if v is not None else None
                           for k, v in data_shapes.items()}

        if _is_notebook:
            display_markdown('### Data Cleaner', raw=True)
            display(data_cleaner, display_id='output_cleaner_info1')

            display_markdown('### Cleaned Train set & Eval set', raw=True)
            display_data = {k: [v] for k, v in data_shapes.items()}
            display(pd.DataFrame(display_data), display_id='output_cleaner_info2')
        else:
            logger.info(f'{self.name} keep {len(selected_features)} columns')

        self.selected_features_ = selected_features
        self.data_cleaner_ = data_cleaner
        self.data_shapes_ = data_shapes

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def cache_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        # 1. Clean Data
        if self.cv and X_eval is not None and y_eval is not None:
            logger.info(f'{self.name} cv enabled, so concat train data and eval data')
            X_train = dex.concat_df([X_train, X_eval], axis=0)
            y_train = dex.concat_df([y_train, y_eval], axis=0)
            X_eval = None
            y_eval = None

        data_cleaner = self.data_cleaner_

        logger.info(f'{self.name} transform train data')
        X_train, y_train = data_cleaner.transform(X_train, y_train)
        self.step_progress('fit_transform train set')

        if X_test is not None:
            logger.info(f'{self.name} transform test data')
            X_test = data_cleaner.transform(X_test)
            self.step_progress('transform X_test')

        if not self.cv:
            if X_eval is None or y_eval is None:
                eval_size = self.experiment.eval_size
                if self.train_test_split_strategy == 'adversarial_validation' and X_test is not None:
                    logger.debug('DriftDetector.train_test_split')
                    detector = self.detector_
                    X_train, X_eval, y_train, y_eval = \
                        detector.train_test_split(X_train, y_train, test_size=eval_size)
                else:
                    if self.task == const.TASK_REGRESSION or dex.is_dask_object(X_train):
                        X_train, X_eval, y_train, y_eval = \
                            dex.train_test_split(X_train, y_train, test_size=eval_size,
                                                 random_state=self.random_state)
                    else:
                        X_train, X_eval, y_train, y_eval = \
                            dex.train_test_split(X_train, y_train, test_size=eval_size,
                                                 random_state=self.random_state, stratify=y_train)
                if self.task != const.TASK_REGRESSION:
                    y_train_uniques = set(y_train.unique()) if hasattr(y_train, 'unique') else set(y_train)
                    y_eval_uniques = set(y_eval.unique()) if hasattr(y_eval, 'unique') else set(y_eval)
                    assert y_train_uniques == y_eval_uniques, \
                        'The classes of `y_train` and `y_eval` must be equal. Try to increase eval_size.'
                self.step_progress('split into train set and eval set')
            else:
                X_eval, y_eval = data_cleaner.transform(X_eval, y_eval)
                self.step_progress('transform eval set')

        selected_features = self.selected_features_
        data_shapes = {'X_train.shape': X_train.shape,
                       'y_train.shape': y_train.shape,
                       'X_eval.shape': None if X_eval is None else X_eval.shape,
                       'y_eval.shape': None if y_eval is None else y_eval.shape,
                       'X_test.shape': None if X_test is None else X_test.shape
                       }
        if dex.exist_dask_object(X_train, y_train, X_eval, y_eval, X_test):
            data_shapes = {k: dex.compute(v) if v is not None else None
                           for k, v in data_shapes.items()}
        if _is_notebook:
            display_markdown('### Data Cleaner (use cache)', raw=True)
            display(data_cleaner, display_id='output_cleaner_info1')

            display_markdown('### Cleaned Train set & Eval set', raw=True)
            display_data = {k: [v] for k, v in data_shapes.items()}
            display(pd.DataFrame(display_data), display_id='output_cleaner_info2')
        else:
            logger.info(f'{self.name} keep {len(selected_features)} columns')

        self.data_shapes_ = data_shapes

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def transform(self, X, y=None, **kwargs):
        return self.data_cleaner_.transform(X, y, **kwargs)

    def get_fitted_params(self):
        dc = self.data_cleaner_

        def get_reason(c):
            if dc is None:
                return 'unknown'

            if dc.dropped_constant_columns_ is not None and c in dc.dropped_constant_columns_:
                return 'constant'
            elif dc.dropped_idness_columns_ is not None and c in dc.dropped_idness_columns_:
                return 'idness'
            elif dc.dropped_duplicated_columns_ is not None and c in dc.dropped_duplicated_columns_:
                return 'duplicated'
            else:
                return 'others'

        params = super().get_fitted_params()
        data_shapes = self.data_shapes_ if self.data_shapes_ is not None else {}
        unselected_features = params.get('unselected_features', [])

        if dc is not None:
            unselected_reason = {f: get_reason(f) for f in unselected_features}
        else:
            unselected_reason = None

        return {**params,
                **data_shapes,
                'unselected_reason': unselected_reason,
                }


class TransformerAdaptorStep(ExperimentStep):
    def __init__(self, experiment, name, transformer_creator, **kwargs):
        assert transformer_creator is not None

        super(TransformerAdaptorStep, self).__init__(experiment, name)

        self.transformer_creator = transformer_creator
        self.transformer_kwargs = kwargs

        for k, v in kwargs.items():
            if not hasattr(self, name):
                try:
                    setattr(self, k, v)
                except:
                    pass

        # fitted
        self.transformer_ = None

    @cache(arg_keys='X_train, y_train, X_test, X_eval, y_eval',
           strategy='transform', transformer='cache_transform',
           attrs_to_restore='transformer_kwargs,transformer_')
    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)

        logger.info(f'{self.name} fit')

        init_kwargs = self.transformer_kwargs.copy()
        if 'task' in init_kwargs.keys():
            init_kwargs['task'] = self.task

        transformer = self.transformer_creator(**init_kwargs)
        transformer.fit(X_train, y_train, **kwargs)
        self.transformer_ = transformer

        return self.cache_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval,
                                    **kwargs)

    def cache_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        logger.info(f'{self.name} cache_transform')

        transformer = self.transformer_
        X_train = transformer.transform(X_train)

        if X_eval is not None:
            X_eval = transformer.transform(X_eval, y_eval)
        if X_test is not None:
            X_test = transformer.transform(X_test)

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def transform(self, X, y=None, **kwargs):
        logger.info(f'{self.name} transform')
        if y is None:
            return self.transformer_.transform(X)
        else:
            return self.transformer_.transform(X, y)


class FeatureGenerationStep(TransformerAdaptorStep):
    def __init__(self, experiment, name,
                 trans_primitives=None,
                 continuous_cols=None,
                 datetime_cols=None,
                 categories_cols=None,
                 latlong_cols=None,
                 text_cols=None,
                 max_depth=1,
                 fix_input=False,
                 feature_selection_args=None):
        drop_cols = []
        if text_cols is not None:
            drop_cols += list(text_cols)
        if latlong_cols is not None:
            drop_cols += list(latlong_cols)

        super(FeatureGenerationStep, self).__init__(experiment, name,
                                                    FeatureGenerationTransformer,
                                                    trans_primitives=trans_primitives,
                                                    fix_input=fix_input,
                                                    continuous_cols=continuous_cols,
                                                    datetime_cols=datetime_cols,
                                                    categories_cols=categories_cols,
                                                    latlong_cols=latlong_cols,
                                                    text_cols=text_cols,
                                                    drop_cols=drop_cols if len(drop_cols) > 0 else None,
                                                    max_depth=max_depth,
                                                    feature_selection_args=feature_selection_args,
                                                    task=None,  # fixed by super
                                                    )


class MulticollinearityDetectStep(FeatureSelectStep):

    def __init__(self, experiment, name):
        super().__init__(experiment, name)

        # fitted
        self.corr_linkage_ = None

    @cache(arg_keys='X_train',
           strategy='transform', transformer='cache_transform',
           attrs_to_restore='input_features_,selected_features_,corr_linkage_')
    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)

        if _is_notebook:
            display_markdown('### Drop features with collinearity', raw=True)

        corr_linkage, remained, dropped = select_by_multicollinearity(X_train)
        self.step_progress('calc correlation')

        if dropped:
            self.selected_features_ = remained

            X_train = X_train[self.selected_features_]
            if X_eval is not None:
                X_eval = X_eval[self.selected_features_]
            if X_test is not None:
                X_test = X_test[self.selected_features_]
            self.step_progress('drop features')
        else:
            self.selected_features_ = None
        self.corr_linkage_ = corr_linkage

        if _is_notebook:
            display(pd.DataFrame([(k, v)
                                  for k, v in self.get_fitted_params().items()],
                                 columns=['key', 'value']),
                    display_id='output_drop_feature_with_collinearity')
        elif logger.is_info_enabled():
            logger.info(f'{self.name} drop {len(dropped)} columns, {len(remained)} kept')

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def get_fitted_params(self):
        return {**super().get_fitted_params(),
                'corr_linkage': self.corr_linkage_,
                }


class DriftDetectStep(FeatureSelectStep):

    def __init__(self, experiment, name, remove_shift_variable, variable_shift_threshold,
                 threshold, remove_size, min_features, num_folds):
        super().__init__(experiment, name)

        self.remove_shift_variable = remove_shift_variable
        self.variable_shift_threshold = variable_shift_threshold

        self.threshold = threshold
        self.remove_size = remove_size if 1.0 > remove_size > 0 else 0.1
        self.min_features = min_features if min_features > 1 else 10
        self.num_folds = num_folds if num_folds > 1 else 5

        # fitted
        self.history_ = None
        self.scores_ = None

    @cache(arg_keys='X_train,X_test',
           strategy='transform', transformer='cache_transform',
           attrs_to_restore='input_features_,selected_features_,history_,scores_')
    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)

        if X_test is not None:
            if _is_notebook:
                display_markdown('### Drift detection', raw=True)

            features, history, scores = dd.feature_selection(X_train, X_test,
                                                             remove_shift_variable=self.remove_shift_variable,
                                                             variable_shift_threshold=self.variable_shift_threshold,
                                                             auc_threshold=self.threshold,
                                                             min_features=self.min_features,
                                                             remove_size=self.remove_size,
                                                             cv=self.num_folds)
            dropped = set(X_train.columns.to_list()) - set(features)
            if dropped:
                self.selected_features_ = features
                X_train = X_train[features]
                X_test = X_test[features]
                if X_eval is not None:
                    X_eval = X_eval[features]
            else:
                self.selected_features_ = None

            self.history_ = history
            self.scores_ = scores

            if _is_notebook:
                display(pd.DataFrame((('no drift features', features),
                                      ('kept/dropped feature count', f'{len(features)}/{len(dropped)}'),
                                      ('history', history),
                                      ('drift score', scores)),
                                     columns=['key', 'value']), display_id='output_drift_detection')
            elif logger.is_info_enabled():
                logger.info(f'{self.name} drop {len(dropped)} columns, {len(features)} kept')

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def get_fitted_params(self):
        return {**super().get_fitted_params(),
                'history': self.history_,
                'scores': self.scores_,
                }


class FeatureImportanceSelectionStep(FeatureSelectStep):
    def __init__(self, experiment, name, strategy, threshold, quantile, number):
        super(FeatureImportanceSelectionStep, self).__init__(experiment, name)

        self.strategy = strategy
        self.threshold = threshold
        self.quantile = quantile
        self.number = number

        # fitted
        self.importances_ = None

    @cache(arg_keys='X_train,y_train',
           strategy='transform', transformer='cache_transform',
           attrs_to_restore='input_features_,selected_features_,importances_')
    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)

        if _is_notebook:
            display_markdown('### Evaluate feature importance', raw=True)

        preprocessor = general_preprocessor(X_train)
        estimator = general_estimator(X_train, task=self.task)
        estimator.fit(preprocessor.fit_transform(X_train, y_train), y_train)
        importances = estimator.feature_importances_
        self.step_progress('training general estimator')

        selected, unselected = \
            select_by_feature_importance(importances, self.strategy,
                                         threshold=self.threshold,
                                         quantile=self.quantile,
                                         number=self.number)

        features = X_train.columns.to_list()
        selected_features = [features[i] for i in selected]
        unselected_features = [features[i] for i in unselected]
        self.step_progress('select by importances')

        if unselected_features:
            X_train = X_train[selected_features]
            if X_eval is not None:
                X_eval = X_eval[selected_features]
            if X_test is not None:
                X_test = X_test[selected_features]

        self.step_progress('drop features')

        if _is_notebook:
            display_markdown('#### feature selection', raw=True)
            is_selected = [i in selected for i in range(len(importances))]
            df = pd.DataFrame(
                zip(X_train.columns.to_list(), importances, is_selected),
                columns=['feature', 'importance', 'selected'])
            df = df.sort_values('importance', axis=0, ascending=False)
            display(df)
        elif logger.is_info_enabled():
            logger.info(f'{self.name} drop {len(unselected_features)} columns, {len(selected_features)} kept')

        self.selected_features_ = selected_features if len(unselected_features) > 0 else None
        self.importances_ = importances

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def get_fitted_params(self):
        return {**super().get_fitted_params(),
                'importances': self.importances_,
                }


class PermutationImportanceSelectionStep(FeatureSelectStep):

    def __init__(self, experiment, name, scorer, estimator_size,
                 strategy, threshold, quantile, number):
        assert scorer is not None

        super().__init__(experiment, name)

        self.scorer = scorer
        self.estimator_size = estimator_size
        self.strategy = strategy
        self.threshold = threshold
        self.quantile = quantile
        self.number = number

        # fixed
        self.importances_ = None

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)

        if _is_notebook:
            display_markdown('### Evaluate permutation importance', raw=True)

        best_trials = hyper_model.get_top_trials(self.estimator_size)
        estimators = [hyper_model.load_estimator(trial.model_file) for trial in best_trials]
        self.step_progress('load estimators')

        if X_eval is None or y_eval is None:
            importances = permutation_importance_batch(estimators, X_train, y_train, self.scorer, n_repeats=5)
        else:
            importances = permutation_importance_batch(estimators, X_eval, y_eval, self.scorer, n_repeats=5)

        if _is_notebook:
            display_markdown('#### importances', raw=True)
            display(pd.DataFrame(
                zip(importances['columns'], importances['importances_mean'], importances['importances_std']),
                columns=['feature', 'importance', 'std']))
            display_markdown('#### feature selection', raw=True)

        # feature_index = np.argwhere(importances.importances_mean < self.threshold)
        # selected_features = [feat for i, feat in enumerate(X_train.columns.to_list()) if i not in feature_index]
        # unselected_features = list(set(X_train.columns.to_list()) - set(selected_features))
        selected, unselected = select_by_feature_importance(importances.importances_mean,
                                                            self.strategy,
                                                            threshold=self.threshold,
                                                            quantile=self.quantile,
                                                            number=self.number)

        if len(selected) > 0:
            selected_features = [importances.columns[i] for i in selected]
            unselected_features = [importances.columns[i] for i in unselected]
        else:
            msg = f'{self.name}: All features will be dropped with importance:{importances.importances_mean},' \
                  f' so drop nothing. Change settings and try again pls.'
            logger.warning(msg)
            selected_features = importances.columns
            unselected_features = []

        self.step_progress('calc importance')

        if unselected_features:
            X_train = X_train[selected_features]
            if X_eval is not None:
                X_eval = X_eval[selected_features]
            if X_test is not None:
                X_test = X_test[selected_features]

        self.step_progress('drop features')

        if _is_notebook:
            display(pd.DataFrame([('Selected', selected_features), ('Unselected', unselected_features)],
                                 columns=['key', 'value']))
        elif logger.is_info_enabled():
            logger.info(f'{self.name} drop {len(unselected_features)} columns, {len(selected_features)} kept')

        self.selected_features_ = selected_features if len(unselected_features) > 0 else None
        self.importances_ = importances

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def get_fitted_params(self):
        return {**super().get_fitted_params(),
                'importances': self.importances_,
                }


class SpaceSearchStep(ExperimentStep):
    def __init__(self, experiment, name, cv=False, num_folds=3):
        super().__init__(experiment, name)

        self.cv = cv
        self.num_folds = num_folds

        # fitted
        self.history_ = None
        self.best_reward_ = None

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)

        if _is_notebook:
            display_markdown('### Pipeline search', raw=True)

        if not dex.is_dask_object(X_eval):
            kwargs['eval_set'] = (X_eval, y_eval)

        model = copy.deepcopy(self.experiment.hyper_model)  # copy from original hyper_model instance
        model.search(X_train, y_train, X_eval, y_eval, cv=self.cv, num_folds=self.num_folds, **kwargs)

        if model.get_best_trial() is None or model.get_best_trial().reward == 0:
            raise RuntimeError('Not found available trial, change experiment settings and try again pls.')

        logger.info(f'{self.name} best_reward: {model.get_best_trial().reward}')

        self.history_ = model.history
        self.best_reward_ = model.get_best_trial().reward

        return model, X_train, y_train, X_test, X_eval, y_eval

    def transform(self, X, y=None, **kwargs):
        return X

    def is_transform_skipped(self):
        return True

    def get_fitted_params(self):
        return {**super().get_fitted_params(),
                'best_reward': self.best_reward_,
                'history': self.history_,
                }


class DaskSpaceSearchStep(SpaceSearchStep):

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        X_train, y_train, X_test, X_eval, y_eval = \
            [v.persist() if dex.is_dask_object(v) else v for v in (X_train, y_train, X_test, X_eval, y_eval)]

        return super().fit_transform(hyper_model, X_train, y_train, X_test, X_eval, y_eval, **kwargs)


class EstimatorBuilderStep(ExperimentStep):
    def __init__(self, experiment, name):
        super().__init__(experiment, name)

        # fitted
        self.estimator_ = None

    def transform(self, X, y=None, **kwargs):
        return X

    def is_transform_skipped(self):
        return True

    def get_fitted_params(self):
        return {**super().get_fitted_params(),
                'estimator': self.estimator_,
                }


class EnsembleStep(EstimatorBuilderStep):
    def __init__(self, experiment, name, scorer=None, ensemble_size=7):
        assert ensemble_size > 1
        super().__init__(experiment, name)

        self.scorer = scorer if scorer is not None else get_scorer('neg_log_loss')
        self.ensemble_size = ensemble_size

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)

        if _is_notebook:
            display_markdown('### Ensemble', raw=True)
        else:
            logger.info('start ensemble')

        best_trials = hyper_model.get_top_trials(self.ensemble_size)
        estimators = [hyper_model.load_estimator(trial.model_file) for trial in best_trials]
        ensemble = self.get_ensemble(estimators, X_train, y_train)

        if all(['oof' in trial.memo.keys() for trial in best_trials]):
            logger.info('ensemble with oofs')
            oofs = self.get_ensemble_predictions(best_trials, ensemble)
            assert oofs is not None
            if hasattr(oofs, 'shape'):
                y_, oofs_ = select_valid_oof(y_train, oofs)
                ensemble.fit(None, y_, oofs_)
            else:
                ensemble.fit(None, y_train, oofs)
        else:
            ensemble.fit(X_eval, y_eval)

        self.estimator_ = ensemble

        if _is_notebook:
            display(ensemble)
        else:
            logger.info(f'ensemble info: {ensemble}')

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    def get_ensemble(self, estimators, X_train, y_train):
        return GreedyEnsemble(self.task, estimators, scoring=self.scorer, ensemble_size=self.ensemble_size)

    def get_ensemble_predictions(self, trials, ensemble):
        oofs = None
        for i, trial in enumerate(trials):
            if 'oof' in trial.memo.keys():
                oof = trial.memo['oof']
                if oofs is None:
                    if len(oof.shape) == 1:
                        oofs = np.zeros((oof.shape[0], len(trials)), dtype=np.float64)
                    else:
                        oofs = np.zeros((oof.shape[0], len(trials), oof.shape[-1]), dtype=np.float64)
                oofs[:, i] = oof

        return oofs


class DaskEnsembleStep(EnsembleStep):
    def get_ensemble(self, estimators, X_train, y_train):
        if dex.exist_dask_object(X_train, y_train):
            predict_kwargs = {}
            if all(['use_cache' in inspect.signature(est.predict).parameters.keys()
                    for est in estimators]):
                predict_kwargs['use_cache'] = False
            return DaskGreedyEnsemble(self.task, estimators, scoring=self.scorer,
                                      ensemble_size=self.ensemble_size,
                                      predict_kwargs=predict_kwargs)

        return super().get_ensemble(estimators, X_train, y_train)

    def get_ensemble_predictions(self, trials, ensemble):
        if isinstance(ensemble, DaskGreedyEnsemble):
            oofs = [trial.memo.get('oof') for trial in trials]
            return oofs if any([oof is not None for oof in oofs]) else None

        return super().get_ensemble_predictions(trials, ensemble)


class FinalTrainStep(EstimatorBuilderStep):
    def __init__(self, experiment, name, retrain_on_wholedata=False):
        super().__init__(experiment, name)

        self.retrain_on_wholedata = retrain_on_wholedata

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)

        if _is_notebook:
            display_markdown('### Load best estimator', raw=True)

        if self.retrain_on_wholedata:
            if _is_notebook:
                display_markdown('#### retrain on whole data', raw=True)
            trial = hyper_model.get_best_trial()
            X_all = dex.concat_df([X_train, X_eval], axis=0)
            y_all = dex.concat_df([y_train, y_eval], axis=0)
            estimator = hyper_model.final_train(trial.space_sample, X_all, y_all, **kwargs)
        else:
            estimator = hyper_model.load_estimator(hyper_model.get_best_trial().model_file)

        self.estimator_ = estimator
        display(estimator)

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval


class PseudoLabelStep(ExperimentStep):
    def __init__(self, experiment, name, estimator_builder,
                 strategy=None, proba_threshold=None, proba_quantile=None, sample_number=None,
                 resplit=False, random_state=None):
        super().__init__(experiment, name)
        assert hasattr(estimator_builder, 'estimator_')

        self.estimator_builder = estimator_builder
        self.strategy = strategy
        self.proba_threshold = proba_threshold
        self.proba_quantile = proba_quantile
        self.sample_number = sample_number
        self.resplit = resplit
        self.random_state = random_state
        self.plot_sample_size = 3000

        # fitted
        self.test_proba_ = None
        self.pseudo_label_stat_ = None

    def transform(self, X, y=None, **kwargs):
        return X

    def is_transform_skipped(self):
        return True

    def fit_transform(self, hyper_model, X_train, y_train, X_test=None, X_eval=None, y_eval=None, **kwargs):
        super().fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval)

        # build estimator
        hyper_model, X_train, y_train, X_test, X_eval, y_eval = \
            self.estimator_builder.fit_transform(hyper_model, X_train, y_train, X_test=X_test,
                                                 X_eval=X_eval, y_eval=y_eval, **kwargs)
        estimator = self.estimator_builder.estimator_

        # start here
        if _is_notebook:
            display_markdown('### Pseudo_label', raw=True)

        X_pseudo = None
        y_pseudo = None
        test_proba = None
        pseudo_label_stat = None
        if self.task in [const.TASK_BINARY, const.TASK_MULTICLASS] and X_test is not None:
            proba = estimator.predict_proba(X_test)
            proba_threshold = self.proba_threshold
            classes = estimator.classes_
            X_pseudo, y_pseudo = sample_by_pseudo_labeling(X_test, classes, proba,
                                                           strategy=self.strategy,
                                                           threshold=self.proba_threshold,
                                                           quantile=self.proba_quantile,
                                                           number=self.sample_number,
                                                           )

            pseudo_label_stat = self.stat_pseudo_label(y_pseudo, classes)
            test_proba = dex.compute(proba)[0] if dex.is_dask_object(proba) else proba
            if test_proba.shape[0] > self.plot_sample_size:
                test_proba, _ = dex.train_test_split(test_proba,
                                                     train_size=self.plot_sample_size,
                                                     random_state=self.random_state)

            proba = test_proba[:, 1]  # fixme
            if _is_notebook:
                display_markdown('### Pseudo label set', raw=True)
                display(pd.DataFrame([(dex.compute(X_pseudo.shape)[0],
                                       dex.compute(y_pseudo.shape)[0],
                                       # len(positive),
                                       # len(negative),
                                       proba_threshold)],
                                     columns=['X_pseudo.shape',
                                              'y_pseudo.shape',
                                              # 'positive samples',
                                              # 'negative samples',
                                              'proba threshold']), display_id='output_presudo_labelings')
            try:
                if _is_notebook:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    # Draw Plot
                    plt.figure(figsize=(8, 4), dpi=80)
                    sns.kdeplot(proba, shade=True, color="g", label="Proba", alpha=.7, bw_adjust=0.01)
                    # Decoration
                    plt.title('Density Plot of Probability', fontsize=22)
                    plt.legend()
                    plt.show()
                # else:
                #     print(proba)
            except:
                print(proba)

        if X_pseudo is not None:
            X_train, y_train, X_eval, y_eval = \
                self.merge_pseudo_label(X_train, y_train, X_eval, y_eval, X_pseudo, y_pseudo)
            if _is_notebook:
                display_markdown('#### Pseudo labeled train set & eval set', raw=True)
                display(pd.DataFrame([(X_train.shape,
                                       y_train.shape,
                                       X_eval.shape if X_eval is not None else None,
                                       y_eval.shape if y_eval is not None else None,
                                       X_test.shape if X_test is not None else None)],
                                     columns=['X_train.shape',
                                              'y_train.shape',
                                              'X_eval.shape',
                                              'y_eval.shape',
                                              'X_test.shape']), display_id='output_cleaner_info2')

        self.test_proba_ = test_proba
        self.pseudo_label_stat_ = pseudo_label_stat

        return hyper_model, X_train, y_train, X_test, X_eval, y_eval

    @staticmethod
    def stat_pseudo_label(y_pseudo, classes):
        stat = OrderedDict()

        if dex.is_dask_object(y_pseudo):
            u = dex.da.unique(y_pseudo, return_counts=True)
            u = dex.compute(u)[0]
        else:
            u = np.unique(y_pseudo, return_counts=True)
        u = {c: n for c, n in zip(*u)}

        for c in classes:
            stat[c] = u[c] if c in u.keys() else 0

        return stat

    def merge_pseudo_label(self, X_train, y_train, X_eval, y_eval, X_pseudo, y_pseudo, **kwargs):
        if self.resplit:
            x_list = [X_train, X_pseudo]
            y_list = [y_train, pd.Series(y_pseudo)]
            if X_eval is not None and y_eval is not None:
                x_list.append(X_eval)
                y_list.append(y_eval)
            X_mix = pd.concat(x_list, axis=0, ignore_index=True)
            y_mix = pd.concat(y_list, axis=0, ignore_index=True)
            if y_mix.dtype != y_train.dtype:
                y_mix = y_mix.astype(y_train.dtype)
            if self.task == const.TASK_REGRESSION:
                stratify = None
            else:
                stratify = y_mix

            eval_size = self.experiment.eval_size
            X_train, X_eval, y_train, y_eval = \
                train_test_split(X_mix, y_mix, test_size=eval_size,
                                 random_state=self.random_state, stratify=stratify)
        else:
            X_train = pd.concat([X_train, X_pseudo], axis=0)
            y_train = pd.concat([y_train, pd.Series(y_pseudo)], axis=0)

        return X_train, y_train, X_eval, y_eval

    def get_fitted_params(self):
        return {**super().get_fitted_params(),
                'test_proba': self.test_proba_,
                'pseudo_label_stat': self.pseudo_label_stat_,
                }


class DaskPseudoLabelStep(PseudoLabelStep):
    def merge_pseudo_label(self, X_train, y_train, X_eval, y_eval, X_pseudo, y_pseudo, **kwargs):
        if not dex.exist_dask_object(X_train, y_train, X_eval, y_eval, X_pseudo, y_pseudo):
            return super().merge_pseudo_label(X_train, y_train, X_eval, y_eval, X_pseudo, y_pseudo, **kwargs)

        if self.resplit:
            x_list = [X_train, X_pseudo]
            y_list = [y_train, y_pseudo]
            if X_eval is not None and y_eval is not None:
                x_list.append(X_eval)
                y_list.append(y_eval)
            X_mix = dex.concat_df(x_list, axis=0)
            y_mix = dex.concat_df(y_list, axis=0)
            # if self.task == const.TASK_REGRESSION:
            #     stratify = None
            # else:
            #     stratify = y_mix

            X_mix = dex.concat_df([X_mix, y_mix], axis=1).reset_index(drop=True)
            y_mix = X_mix.pop(y_mix.name)

            eval_size = self.experiment.eval_size
            X_train, X_eval, y_train, y_eval = \
                dex.train_test_split(X_mix, y_mix, test_size=eval_size, random_state=self.random_state)
        else:
            X_train = dex.concat_df([X_train, X_pseudo], axis=0)
            y_train = dex.concat_df([y_train, y_pseudo], axis=0)

            # align divisions
            X_train = dex.concat_df([X_train, y_train], axis=1)
            y_train = X_train.pop(y_train.name)

        return X_train, y_train, X_eval, y_eval


class SteppedExperiment(Experiment):
    def __init__(self, steps, *args, **kwargs):
        assert isinstance(steps, (tuple, list)) and all([isinstance(step, ExperimentStep) for step in steps])
        super(SteppedExperiment, self).__init__(*args, **kwargs)

        if logger.is_info_enabled():
            names = [step.name for step in steps]
            logger.info(f'create experiment with {names}')
        self.steps = steps

    def train(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, **kwargs):
        for step in self.steps:
            if X_test is not None and X_train.columns.to_list() != X_test.columns.to_list():
                logger.warning(f'X_train{X_train.columns.to_list()} and X_test{X_test.columns.to_list()}'
                               f' have different columns before {step.name}, try fix it.')
                X_test = X_test[X_train.columns]
            if X_eval is not None and X_train.columns.to_list() != X_eval.columns.to_list():
                logger.warning(f'X_train{X_train.columns.to_list()} and X_eval{X_eval.columns.to_list()}'
                               f' have different columns before {step.name}, try fix it.')
                X_eval = X_eval[X_train.columns]

            X_train, y_train, X_test, X_eval, y_eval = \
                [v.persist() if dex.is_dask_object(v) else v for v in (X_train, y_train, X_test, X_eval, y_eval)]

            if logger.is_debug_enabled():
                logger.debug(f'fit_transform {step.name} with {X_train.columns.to_list()}')
            else:
                logger.info(f'fit_transform {step.name}')

            self.step_start(step.name)
            try:
                hyper_model, X_train, y_train, X_test, X_eval, y_eval = \
                    step.fit_transform(hyper_model, X_train, y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval,
                                       **kwargs)
                self.step_end(output=step.get_fitted_params())
                if step.status_ is None:
                    step.status_ = True
            except Exception as e:
                self.step_break(error=e)
                if step.status_ is None:
                    step.status_ = False
                raise e

        estimator = self.to_estimator(self.steps)
        self.hyper_model = hyper_model

        return estimator

    def get_step(self, name):
        for step in self.steps:
            if step.name == name:
                return step

        raise ValueError(f'Not found step "{name}"')

    @staticmethod
    def to_estimator(steps):
        last_step = steps[-1]
        assert hasattr(last_step, 'estimator_')

        pipeline_steps = [(step.name, step) for step in steps if not step.is_transform_skipped()]

        if len(pipeline_steps) > 0:
            pipeline_steps += [('estimator', last_step.estimator_)]
            estimator = Pipeline(pipeline_steps)
            if logger.is_info_enabled():
                names = [step[0] for step in pipeline_steps]
                logger.info(f'trained experiment pipeline: {names}')
        else:
            estimator = last_step.estimator_
            if logger.is_info_enabled():
                logger.info(f'trained experiment estimator:\n{estimator}')

        return estimator


class CompeteExperiment(SteppedExperiment):
    """
    A powerful experiment strategy for AutoML with a set of advanced features.

    There are still many challenges in the machine learning modeling process for tabular data, such as imbalanced data,
    data drift, poor generalization ability, etc.  This challenges cannot be completely solved by pipeline search,
    so we introduced in HyperNets a more powerful tool is `CompeteExperiment`. `CompeteExperiment` is composed of a series
    of steps and *Pipeline Search* is just one step. It also includes advanced steps such as data cleaning,
    data drift handling, two-stage search, ensemble etc.
    """

    def __init__(self, hyper_model, X_train, y_train, X_eval=None, y_eval=None, X_test=None,
                 eval_size=DEFAULT_EVAL_SIZE,
                 train_test_split_strategy=None,
                 cv=True, num_folds=3,
                 task=None,
                 id=None,
                 callbacks=None,
                 random_state=9527,
                 scorer=None,
                 data_cleaner_args=None,
                 feature_generation=False,
                 feature_generation_trans_primitives=None,
                 # feature_generation_fix_input=False,
                 feature_generation_max_depth=1,
                 feature_generation_categories_cols=None,
                 feature_generation_continuous_cols=None,
                 feature_generation_datetime_cols=None,
                 feature_generation_latlong_cols=None,
                 feature_generation_text_cols=None,
                 # feature_generation_feature_selection_args=None,
                 collinearity_detection=False,
                 drift_detection=True,
                 drift_detection_remove_shift_variable=True,
                 drift_detection_variable_shift_threshold=0.7,
                 drift_detection_threshold=0.7,
                 drift_detection_remove_size=0.1,
                 drift_detection_min_features=10,
                 drift_detection_num_folds=5,
                 feature_selection=False,
                 feature_selection_strategy=None,
                 feature_selection_threshold=None,
                 feature_selection_quantile=None,
                 feature_selection_number=None,
                 ensemble_size=20,
                 feature_reselection=False,
                 feature_reselection_estimator_size=10,
                 feature_reselection_strategy=None,
                 feature_reselection_threshold=1e-5,
                 feature_reselection_quantile=None,
                 feature_reselection_number=None,
                 pseudo_labeling=False,
                 pseudo_labeling_strategy=None,
                 pseudo_labeling_proba_threshold=None,
                 pseudo_labeling_proba_quantile=None,
                 pseudo_labeling_sample_number=None,
                 pseudo_labeling_resplit=False,
                 retrain_on_wholedata=False,
                 log_level=None,
                 **kwargs):
        """
        Parameters
        ----------
        hyper_model : hypernets.model.HyperModel
            A `HyperModel` instance
        X_train : Pandas or Dask DataFrame
            Feature data for training
        y_train : Pandas or Dask Series
            Target values for training
        X_eval : (Pandas or Dask DataFrame) or None
            (default=None), Feature data for evaluation
        y_eval : (Pandas or Dask Series) or None, (default=None)
            Target values for evaluation
        X_test : (Pandas or Dask Series) or None, (default=None)
            Unseen data without target values for semi-supervised learning
        eval_size : float or int, (default=None)
            Only valid when ``X_eval`` or ``y_eval`` is None. If float, should be between 0.0 and 1.0 and represent
            the proportion of the dataset to include in the eval split. If int, represents the absolute number of
            test samples. If None, the value is set to the complement of the train size.
        train_test_split_strategy : *'adversarial_validation'* or None, (default=None)
            Only valid when ``X_eval`` or ``y_eval`` is None. If None, use eval_size to split the dataset,
            otherwise use adversarial validation approach.
        cv : bool, (default=True)
            If True, use cross-validation instead of evaluation set reward to guide the search process
        num_folds : int, (default=3)
            Number of cross-validated folds, only valid when cv is true
        task : str or None, (default=None)
            Task type(*binary*, *multiclass* or *regression*).
            If None, inference the type of task automatically
        callbacks : list of callback functions or None, (default=None)
            List of callback functions that are applied at each experiment step. See `hypernets.experiment.ExperimentCallback`
            for more information.
        random_state : int or RandomState instance, (default=9527)
            Controls the shuffling applied to the data before applying the split
        scorer : str, callable or None, (default=None)
            Scorer to used for feature importance evaluation and ensemble. It can be a single string
            (see [get_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html))
            or a callable (see [make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)).
            If None, exception will occur.
        data_cleaner_args : dict, (default=None)
            dictionary of parameters to initialize the `DataCleaner` instance. If None, `DataCleaner` will initialized
            with default values.
        feature_generation : bool (default False),
            Whether to enable feature generation.
        feature_generation_trans_primitives: list (default None)
            FeatureTools transform primitives list.
        feature_generation_categories_cols: list (default None),
            Column names to generate new features as FeatureTools Categorical variables.
        feature_generation_continuous_cols: list (default detected from X_train),
            Column names to generate new features as FeatureTools Numeric variables.
        feature_generation_datetime_cols: list (default detected from X_train),
            Column names to generate new features as FeatureTools Datetime variables.
        feature_generation_latlong_cols: list (default None),
            Column names to generate new features as FeatureTools LatLong variables.
        feature_generation_text_cols: list (default None),
            Column names to generate new features as FeatureTools Text(NaturalLanguage) variables.
        collinearity_detection :  bool, (default=False)
            Whether to clear multicollinearity features
        drift_detection : bool,(default=True)
            Whether to enable data drift detection and processing. Only valid when *X_test* is provided. Concept drift
            in the input data is one of the main challenges. Over time, it will worsen the performance of model on new
            data. We introduce an adversarial validation approach to concept drift problems. This approach will detect
            concept drift and identify the drifted features and process them automatically.
        drift_detection_remove_shift_variable : bool, (default=True)
        drift_detection_variable_shift_threshold : float, (default=0.7)
        drift_detection_threshold : float, (default=0.7)
        drift_detection_remove_size : float, (default=0.1)
        drift_detection_min_features : int, (default=10)
        drift_detection_num_folds : int, (default=5)
        feature_selection: bool, (default=False)
            Whether to select features by *feature_importances_*.
        feature_selection_strategy : str, (default='threshold')
            Strategy to select features(*threshold*, *number* or *quantile*).
        feature_selection_threshold : float, (default=0.1)
            Confidence threshold of feature_importance. Only valid when *feature_selection_strategy* is 'threshold'.
        feature_selection_quantile:
            Confidence quantile of feature_importance. Only valid when *feature_selection_strategy* is 'quantile'.
        feature_selection_number:
            Expected feature number to keep. Only valid when *feature_selection_strategy* is 'number'.
        feature_reselection : bool, (default=True)
            Whether to enable two stage feature selection with permutation importance.
        feature_reselection_estimator_size : int, (default=10)
            The number of estimator to evaluate feature importance. Only valid when *feature_reselection* is True.
        feature_reselection_strategy : str, (default='threshold')
            Strategy to reselect features(*threshold*, *number* or *quantile*).
        feature_reselection_threshold : float, (default=1e-5)
            Confidence threshold of the mean permutation importance. Only valid when *feature_reselection_strategy* is 'threshold'.
        feature_reselection_quantile:
            Confidence quantile of feature_importance. Only valid when *feature_reselection_strategy* is 'quantile'.
        feature_reselection_number:
            Expected feature number to keep. Only valid when *feature_reselection_strategy* is 'number'.
        ensemble_size : int, (default=20)
            The number of estimator to ensemble. During the AutoML process, a lot of models will be generated with different
            preprocessing pipelines, different models, and different hyperparameters. Usually selecting some of the models
            that perform well to ensemble can obtain better generalization ability than just selecting the single best model.
        pseudo_labeling : bool, (default=False)
            Whether to enable pseudo labeling. Pseudo labeling is a semi-supervised learning technique, instead of manually
            labeling the unlabelled data, we give approximate labels on the basis of the labelled data. Pseudo-labeling can
            sometimes improve the generalization capabilities of the model.
        pseudo_labeling_strategy : str, (default='threshold')
            Strategy to sample pseudo labeling data(*threshold*, *number* or *quantile*).
        pseudo_labeling_proba_threshold : float, (default=0.8)
            Confidence threshold of pseudo-label samples. Only valid when *pseudo_labeling_strategy* is 'threshold'.
        pseudo_labeling_proba_quantile:
            Confidence quantile of pseudo-label samples. Only valid when *pseudo_labeling_strategy* is 'quantile'.
        pseudo_labeling_sample_number:
            Excepted number to sample per class. Only valid when *pseudo_labeling_strategy* is 'number'.
        pseudo_labeling_resplit : bool, (default=False)
            Whether to re-split the training set and evaluation set after adding pseudo-labeled data. If False, the
            pseudo-labeled data is only appended to the training set. Only valid when *pseudo_labeling* is True.
        retrain_on_wholedata : bool, (default=False)
            Whether to retrain the model with whole data after the search is completed.
        log_level : int, str, or None (default=None),
            Level of logging, possible values:
                -logging.CRITICAL
                -logging.FATAL
                -logging.ERROR
                -logging.WARNING
                -logging.WARN
                -logging.INFO
                -logging.DEBUG
                -logging.NOTSET
        kwargs :

        """
        steps = []
        two_stage = False
        enable_dask = dex.exist_dask_object(X_train, y_train, X_test, X_eval, y_eval)

        if enable_dask:
            search_cls, ensemble_cls, pseudo_cls = SpaceSearchStep, DaskEnsembleStep, DaskPseudoLabelStep
        else:
            search_cls, ensemble_cls, pseudo_cls = SpaceSearchStep, EnsembleStep, PseudoLabelStep

        # data clean
        steps.append(DataCleanStep(self, 'data_clean',
                                   data_cleaner_args=data_cleaner_args, cv=cv,
                                   train_test_split_strategy=train_test_split_strategy,
                                   random_state=random_state))
        # feature generation
        if feature_generation:
            steps.append(FeatureGenerationStep(
                self, 'feature_generation',
                trans_primitives=feature_generation_trans_primitives,
                max_depth=feature_generation_max_depth,
                continuous_cols=feature_generation_continuous_cols,
                datetime_cols=feature_generation_datetime_cols,
                categories_cols=feature_generation_categories_cols,
                latlong_cols=feature_generation_latlong_cols,
                text_cols=feature_generation_text_cols,
            ))

        # select by collinearity
        if collinearity_detection:
            steps.append(MulticollinearityDetectStep(self, 'collinearity_detection'))

        # drift detection
        if drift_detection:
            steps.append(DriftDetectStep(self, 'drift_detection',
                                         remove_shift_variable=drift_detection_remove_shift_variable,
                                         variable_shift_threshold=drift_detection_variable_shift_threshold,
                                         threshold=drift_detection_threshold,
                                         remove_size=drift_detection_remove_size,
                                         min_features=drift_detection_min_features,
                                         num_folds=drift_detection_num_folds))
        # feature selection by importance
        if feature_selection:
            steps.append(FeatureImportanceSelectionStep(
                self, 'feature_selection',
                strategy=feature_selection_strategy,
                threshold=feature_selection_threshold,
                quantile=feature_selection_quantile,
                number=feature_selection_number))

        # first-stage search
        steps.append(search_cls(self, 'space_search', cv=cv, num_folds=num_folds))

        # pseudo label
        if pseudo_labeling and task != const.TASK_REGRESSION:
            if ensemble_size is not None and ensemble_size > 1:
                estimator_builder = ensemble_cls(self, 'pseudo_ensemble', scorer=scorer, ensemble_size=ensemble_size)
            else:
                estimator_builder = FinalTrainStep(self, 'pseudo_train', retrain_on_wholedata=retrain_on_wholedata)
            step = pseudo_cls(self, 'pseudo_labeling',
                              estimator_builder=estimator_builder,
                              strategy=pseudo_labeling_strategy,
                              proba_threshold=pseudo_labeling_proba_threshold,
                              proba_quantile=pseudo_labeling_proba_quantile,
                              sample_number=pseudo_labeling_sample_number,
                              resplit=pseudo_labeling_resplit,
                              random_state=random_state)
            steps.append(step)
            two_stage = True

        # importance selection
        if feature_reselection:
            step = PermutationImportanceSelectionStep(
                self, 'feature_reselection',
                scorer=scorer,
                estimator_size=feature_reselection_estimator_size,
                strategy=feature_reselection_strategy,
                threshold=feature_reselection_threshold,
                quantile=feature_reselection_quantile,
                number=feature_reselection_number)
            steps.append(step)
            two_stage = True

        # two-stage search
        if two_stage:
            steps.append(search_cls(self, 'two_stage_search', cv=cv, num_folds=num_folds))

        # final train
        if ensemble_size is not None and ensemble_size > 1:
            last_step = ensemble_cls(self, 'final_ensemble', scorer=scorer, ensemble_size=ensemble_size)
        else:
            last_step = FinalTrainStep(self, 'final_train', retrain_on_wholedata=retrain_on_wholedata)
        steps.append(last_step)

        # ignore warnings
        import warnings
        warnings.filterwarnings('ignore')

        if log_level is not None:
            _set_log_level(log_level)

        self.run_kwargs = kwargs
        super(CompeteExperiment, self).__init__(steps,
                                                hyper_model, X_train, y_train, X_eval=X_eval, y_eval=y_eval,
                                                X_test=X_test, eval_size=eval_size, task=task,
                                                id=id,
                                                callbacks=callbacks,
                                                random_state=random_state)

    def train(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, **kwargs):
        if _is_notebook:
            display_markdown('### Input Data', raw=True)

            if dex.exist_dask_object(X_train, y_train, X_test, X_eval, y_eval):
                display_data = (dex.compute(X_train.shape)[0],
                                dex.compute(y_train.shape)[0],
                                dex.compute(X_eval.shape)[0] if X_eval is not None else None,
                                dex.compute(y_eval.shape)[0] if y_eval is not None else None,
                                dex.compute(X_test.shape)[0] if X_test is not None else None,
                                self.task if self.task == const.TASK_REGRESSION
                                else f'{self.task}({dex.compute(y_train.nunique())[0]})')
            else:
                display_data = (X_train.shape,
                                y_train.shape,
                                X_eval.shape if X_eval is not None else None,
                                y_eval.shape if y_eval is not None else None,
                                X_test.shape if X_test is not None else None,
                                self.task if self.task == const.TASK_REGRESSION
                                else f'{self.task}({y_train.nunique()})')
            display(pd.DataFrame([display_data],
                                 columns=['X_train.shape',
                                          'y_train.shape',
                                          'X_eval.shape',
                                          'y_eval.shape',
                                          'X_test.shape',
                                          'Task', ]), display_id='output_intput')

            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y = le.fit_transform(y_train.dropna())
            # Draw Plot
            plt.figure(figsize=(8, 4), dpi=80)
            sns.distplot(y, kde=False, color="g", label="y")
            # Decoration
            plt.title('Distribution of y', fontsize=12)
            plt.legend()
            plt.show()

        return super().train(hyper_model, X_train, y_train, X_test, X_eval, y_eval, **kwargs)

    def run(self, **kwargs):
        run_kwargs = {**self.run_kwargs, **kwargs}
        return super().run(**run_kwargs)


def evaluate_oofs(hyper_model, ensemble_estimator, y_train, metrics):
    from hypernets.tabular.lifelong_learning import select_valid_oof
    from hypernets.tabular.metrics import calc_score
    trials = hyper_model.get_top_trials(ensemble_estimator.ensemble_size)
    if all(['oof' in trial.memo.keys() for trial in trials]):
        oofs = None
        for i, trial in enumerate(trials):
            if 'oof' in trial.memo.keys():
                oof = trial.memo['oof']
                if oofs is None:
                    if len(oof.shape) == 1:
                        oofs = np.zeros((oof.shape[0], len(trials)), dtype=np.float64)
                    else:
                        oofs = np.zeros((oof.shape[0], len(trials), oof.shape[-1]), dtype=np.float64)
                oofs[:, i] = oof
        y_, oofs_ = select_valid_oof(y_train, oofs)
        proba = ensemble_estimator.predictions2predict_proba(oofs_)
        pred = ensemble_estimator.predictions2predict(oofs_)
        scores = calc_score(y_, pred, proba, metrics)
        return scores
    else:
        print('No oof data')
        return None
