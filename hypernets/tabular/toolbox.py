# -*- coding:utf-8 -*-
"""

"""
import copy
import gc as gc_
import math
from functools import partial

import numpy as np
import pandas as pd
import psutil
from sklearn import model_selection as sk_ms, preprocessing as sk_pre, \
    decomposition as sk_dec, utils as sk_utils, inspection, pipeline

from hypernets.core import randint
from hypernets.utils import logging, const
from . import collinearity as collinearity_
from . import column_selector as column_selector_
from . import data_cleaner as data_cleaner_
from . import data_hasher as data_hasher_
from . import dataframe_mapper as dataframe_mapper_
from . import drift_detection as drift_detection_
from . import ensemble as ensemble_
from . import estimator_detector as estimator_detector_
from . import feature_generators as feature_generators_
from . import metrics as metrics_
from . import pseudo_labeling as pseudo_labeling_
from . import sklearn_ex as sk_ex  # NOQA,  register customized transformer
from ._base import ToolboxMeta, register_transformer
from .cfg import TabularCfg as c

try:
    import lightgbm

    lightgbm_installed = True
except ImportError:
    lightgbm_installed = False

logger = logging.get_logger(__name__)


class ToolBox(metaclass=ToolboxMeta):
    STRATEGY_THRESHOLD = 'threshold'
    STRATEGY_QUANTILE = 'quantile'
    STRATEGY_NUMBER = 'number'

    acceptable_types = (np.ndarray, pd.DataFrame, pd.Series)

    @classmethod
    def accept(cls, *args):
        def is_acceptable(x):
            if x is None:
                return True
            if isinstance(x, type) and x in cls.acceptable_types:
                return True
            if type(x) in cls.acceptable_types:
                return True

            return False

        return all(map(is_acceptable, args))

    @staticmethod
    def get_shape(X, allow_none=False):
        if allow_none and X is None:
            return None
        else:
            return X.shape

    @staticmethod
    def memory_total():
        mem = psutil.virtual_memory()
        return mem.total

    @staticmethod
    def memory_free():
        mem = psutil.virtual_memory()
        return mem.available

    @staticmethod
    def memory_usage(*data):
        usage = 0
        for x in data:
            if isinstance(x, pd.DataFrame):
                usage += x.memory_usage().sum()
            elif isinstance(x, pd.Series):
                usage += x.memory_usage()
            elif isinstance(x, np.ndarray):
                usage += x.nbytes
            else:
                pass  # ignore
        return usage

    @staticmethod
    def to_local(*data):
        return data

    @staticmethod
    def gc():
        gc_.collect()
        gc_.collect()

    @staticmethod
    def from_local(*data):
        return data

    @staticmethod
    def load_data(data_path, *, reset_index=False, reader_mapping=None, **kwargs):
        import os.path as path
        import glob

        if reader_mapping is None:
            reader_mapping = {
                'csv': partial(pd.read_csv, low_memory=False),
                'txt': partial(pd.read_csv, low_memory=False),
                'parquet': pd.read_parquet,
                'par': pd.read_parquet,
                'json': pd.read_json,
                'pkl': pd.read_pickle,
                'pickle': pd.read_pickle,
            }

        def get_file_format(file_path):
            return path.splitext(file_path)[-1].lstrip('.')

        def get_file_format_by_glob(data_pattern):
            for f in glob.glob(data_pattern, recursive=True):
                fmt_ = get_file_format(f)
                if fmt_ in reader_mapping.keys():
                    return fmt_
            return None

        if glob.has_magic(data_path):
            fmt = get_file_format_by_glob(data_path)
        elif not path.exists(data_path):
            raise ValueError(f'Not found path {data_path}')
        elif path.isdir(data_path):
            path_pattern = f'{data_path}*' if data_path.endswith(path.sep) else f'{data_path}{path.sep}*'
            fmt = get_file_format_by_glob(path_pattern)
        else:
            fmt = path.splitext(data_path)[-1].lstrip('.')

        if fmt not in reader_mapping.keys():
            raise ValueError(f'Not supported data format{fmt}')
        fn = reader_mapping[fmt]
        df = fn(data_path, **kwargs)

        if reset_index:
            df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def parquet():
        from .persistence import ParquetPersistence
        return ParquetPersistence()

    @staticmethod
    def unique(y):
        if hasattr(y, 'unique'):
            uniques = set(y.unique())
        else:
            uniques = set(y)
        return uniques

    # @staticmethod
    # def unique_array(ar, return_index=False, return_inverse=False, return_counts=False, axis=None):
    #     return np.unique(ar, return_index=return_index, return_inverse=return_inverse,
    #                      return_counts=return_counts, axis=axis)

    @staticmethod
    def nunique_df(df):
        return df.nunique(dropna=True)

    @staticmethod
    def value_counts(ar):
        return pd.Series(ar).value_counts().to_dict()

    @staticmethod
    def select_df(df, indices):
        """
        Select dataframe by row indices.
        """
        return df.iloc[indices]

    @staticmethod
    def select_1d(arr, indices):
        """
        Select by indices from the first axis(0).
        """
        if hasattr(arr, 'iloc'):
            return arr.iloc[indices]
        else:
            return arr[indices]

    @staticmethod
    def collapse_last_dim(arr, keep_dim=True):
        """
        Collapse the last dimension
        :param arr: data array
        :param keep_dim: keep the last dim as one or not
        :return:
        """

        def _collapse(x):
            return x

        fn = np.vectorize(_collapse, otypes=[object], signature='(m)->()')
        t = fn(arr)
        if keep_dim:
            shape = arr.shape[:-1] + (1,)
            t = t.reshape(shape)
        return t

    @classmethod
    def hstack_array(cls, arrs):
        arrs = [a.reshape(-1, 1) if a.ndim == 1 else a for a in arrs]
        return cls.stack_array(arrs, axis=1)

    @classmethod
    def vstack_array(cls, arrs):
        return cls.stack_array(arrs, axis=0)

    @staticmethod
    def stack_array(arrs, axis=0):
        assert axis in (0, 1)
        ndims = set([len(a.shape) for a in arrs])
        if len(ndims) > 1:
            assert ndims == {1, 2}
            assert all([len(a.shape) == 1 or a.shape[1] == 1 for a in arrs])
            arrs = [a.reshape(-1, 1) if len(a.shape) == 1 else a for a in arrs]
        axis = min(axis, min([len(a.shape) for a in arrs]) - 1)
        assert axis >= 0

        return np.concatenate(arrs, axis=axis)

    @staticmethod
    def take_array(arr, indices, axis=None):
        return np.take(arr, indices=indices, axis=axis)

    @staticmethod
    def array_to_df(arr, *, columns=None, index=None, meta=None):
        return pd.DataFrame(arr, columns=columns, index=index)

    @staticmethod
    def df_to_array(df):
        return df.values

    @staticmethod
    def concat_df(dfs, axis=0, repartition=False, random_state=9527, **kwargs):
        header = dfs[0]
        assert isinstance(header, (pd.DataFrame, pd.Series))

        def to_pd_type(t):
            if isinstance(t, (pd.DataFrame, pd.Series)):
                return t
            elif isinstance(header, pd.Series):
                return pd.Series(t, name=header.name)
            else:
                return pd.DataFrame(t, columns=header.columns)

        dfs = [header] + [to_pd_type(df) for df in dfs[1:]]
        df = pd.concat(dfs, axis=axis, **kwargs)
        if repartition:
            df = df.sample(frac=1.0, random_state=random_state)
        return df

    @staticmethod
    def reset_index(df):
        return df.reset_index(drop=True)

    @staticmethod
    def merge_oof(oofs):
        """
        :param oofs: list of tuple(idx,proba)
        :return: merged proba
        """
        row_count = sum(map(lambda x: len(x[0]), oofs))
        max_idx = max(map(lambda x: np.max(x[0]), oofs))
        if max_idx >= row_count:
            row_count = max_idx + 1

        proba = oofs[0][1]
        if len(proba.shape) == 1:
            r = np.full(row_count, np.nan, proba.dtype)
        else:
            r = np.full((row_count, proba.shape[-1]), np.nan, proba.dtype)

        for idx, proba in oofs:
            r[idx] = proba

        return r

    @staticmethod
    def select_valid_oof(y, oof):
        if len(oof.shape) == 1:
            idx = np.argwhere(~np.isnan(oof[:])).ravel()
        elif len(oof.shape) == 2:
            idx = np.argwhere(~np.isnan(oof[:, 0])).ravel()
        elif len(oof.shape) == 3:
            idx = np.argwhere(~np.isnan(oof[:, 0, 0])).ravel()
        else:
            raise ValueError(f'Unsupported shape:{oof.shape}')

        if hasattr(y, 'iloc'):
            return y.iloc[idx], oof[idx]
        else:
            return y[idx], oof[idx]

    @classmethod
    def infer_task_type(cls, y, excludes=None):
        assert excludes is None or isinstance(excludes, (list, tuple, set))

        if len(y.shape) > 1 and y.shape[-1] > 1:
            labels = list(range(y.shape[-1]))
            task = const.TASK_MULTILABEL  # 'multilable'
            return task, labels

        uniques = cls.unique(y)
        if uniques.__contains__(np.nan):
            uniques.remove(np.nan)
        if excludes is not None and len(excludes) > 0:
            uniques -= set(excludes)
        n_unique = len(uniques)
        labels = []

        if n_unique == 2:
            logger.info(f'2 class detected, {uniques}, so inferred as a [binary classification] task')
            task = const.TASK_BINARY  # TASK_BINARY
            labels = sorted(uniques)
        else:
            if str(y.dtype).find('float') >= 0:
                logger.info(f'Target column type is {y.dtype}, so inferred as a [regression] task.')
                task = const.TASK_REGRESSION
            else:
                if n_unique > 1000:
                    if str(y.dtype).find('int') >= 0:
                        logger.info('The number of classes exceeds 1000 and column type is {y.dtype}, '
                                    'so inferred as a [regression] task ')
                        task = const.TASK_REGRESSION
                    else:
                        raise ValueError('The number of classes exceeds 1000, please confirm whether '
                                         'your predict target is correct ')
                else:
                    logger.info(f'{n_unique} class detected, inferred as a [multiclass classification] task')
                    task = const.TASK_MULTICLASS
                    labels = sorted(uniques)
        return task, labels

    @staticmethod
    def mean_oof(probas):
        return np.mean(probas, axis=0)

    @staticmethod
    def fix_binary_predict_proba_result(proba):
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T
        elif proba.shape[1] == 1:
            proba = np.hstack([1 - proba, proba])

        return proba

    @classmethod
    def general_preprocessor(cls, X, y=None):
        cs = cls.column_selector
        tfs = cls.transformers

        cat_steps = [('imputer_cat', tfs['SimpleImputer'](strategy='constant', fill_value='')),
                     ('encoder', tfs['SafeOrdinalEncoder']())]
        num_steps = [('imputer_num', tfs['SimpleImputer'](strategy='mean')),
                     ('scaler', tfs['StandardScaler']())]

        cat_transformer = tfs['Pipeline'](steps=cat_steps)
        num_transformer = tfs['Pipeline'](steps=num_steps)

        preprocessor = tfs['DataFrameMapper'](
            features=[(cs.column_object_category_bool, cat_transformer),
                      (cs.column_number_exclude_timedelta, num_transformer)],
            input_df=True,
            df_out=True)
        return preprocessor

    @classmethod
    def general_estimator(cls, X, y=None, estimator=None, task=None):
        def default_gbm(task_):
            est_cls = lightgbm.LGBMRegressor if task_ == const.TASK_REGRESSION else lightgbm.LGBMClassifier
            return est_cls(n_estimators=50,
                           num_leaves=15,
                           max_depth=5,
                           subsample=0.5,
                           subsample_freq=1,
                           colsample_bytree=0.8,
                           reg_alpha=1,
                           reg_lambda=1,
                           importance_type='gain',
                           random_state=randint(),
                           verbose=-1)

        def default_dt(task_):
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            est_cls = DecisionTreeRegressor if task_ == const.TASK_REGRESSION else DecisionTreeClassifier
            return est_cls(min_samples_leaf=20, min_impurity_decrease=0.01, random_state=randint())

        def default_rf(task_):
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            est_cls = RandomForestRegressor if task_ == const.TASK_REGRESSION else RandomForestClassifier
            return est_cls(min_samples_leaf=20, min_impurity_decrease=0.01, random_state=randint())

        if estimator is None:
            estimator = 'gbm' if lightgbm_installed else 'rf'
        if task is None:
            assert y is not None, '"y" or "task" is required.'
            task = cls.infer_task_type(y)

        if estimator == 'gbm':
            estimator_ = default_gbm(task)
        elif estimator == 'dt':
            estimator_ = default_dt(task)
        elif estimator == 'rf':
            estimator_ = default_rf(task)
        else:
            estimator_ = copy.deepcopy(estimator)

        return estimator_

    @staticmethod
    def permutation_importance(estimator, X, y, *,
                               scoring=None, n_repeats=5, n_jobs=None,
                               random_state=None, sample_weight=None, max_samples=1.0):
        """
        see: sklearn.inspection.permutation_importance
        """

        if n_jobs is None and c.joblib_njobs is not None and c.joblib_njobs > 0:
            n_jobs = c.joblib_njobs

        if hasattr(estimator, 'permutation_importance'):
            importance = estimator.permutation_importance(X, y,
                                                          scoring=scoring, n_repeats=n_repeats, n_jobs=n_jobs,
                                                          random_state=random_state)
        else:
            importance = inspection.permutation_importance(estimator, X, y,
                                                           scoring=scoring, n_repeats=n_repeats, n_jobs=n_jobs,
                                                           random_state=random_state)
        return importance

    @classmethod
    def permutation_importance_batch(cls, estimators, X, y, scoring=None, n_repeats=5,
                                     n_jobs=None, random_state=None):
        """Evaluate the importance of features of a set of estimators

        Parameters
        ----------
        estimators : list
            A set of estimators that has already been :term:`fitted` and is compatible
            with :term:`scorer`.

        X : ndarray or DataFrame, shape (n_samples, n_features)
            Data on which permutation importance will be computed.

        y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
            Targets for supervised or `None` for unsupervised.

        scoring : string, callable or None, default=None
            Scorer to use. It can be a single
            string (see :ref:`scoring_parameter`) or a callable (see
            :ref:`scoring`). If None, the estimator's default scorer is used.

        n_repeats : int, default=5
            Number of times to permute a feature.

        n_jobs : int or None, default=None
            The number of jobs to use for the computation.
            `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
            `-1` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        random_state : int, RandomState instance, or None, default=None
            Pseudo-random number generator to control the permutations of each
            feature. See :term:`random_state`.

        Returns
        -------
        result : Bunch
            Dictionary-like object, with attributes:

            importances_mean : ndarray, shape (n_features, )
                Mean of feature importance over `n_repeats`.
            importances_std : ndarray, shape (n_features, )
                Standard deviation over `n_repeats`.
            importances : ndarray, shape (n_features, n_repeats)
                Raw permutation importance scores.
        """
        importances = []

        X_shape = cls.get_shape(X)
        if X_shape[0] > c.permutation_importance_sample_limit:
            if logger.is_info_enabled():
                logger.info(f'{X_shape[0]} rows data found, sample to {c.permutation_importance_sample_limit}')
            frac = c.permutation_importance_sample_limit / X_shape[0]
            X, _, y, _ = cls.train_test_split(X, y, train_size=frac, random_state=random_state)

        if isinstance(n_jobs, int) and n_jobs <= 0:
            n_jobs = None  # higher performance than -1

        for i, est in enumerate(estimators):
            if logger.is_info_enabled():
                logger.info(f'score permutation importance by estimator {i}/{len(estimators)}')
            importance = cls.permutation_importance(est, X.copy(), y.copy(),
                                                    scoring=scoring, n_repeats=n_repeats, n_jobs=n_jobs,
                                                    random_state=random_state)
            importances.append(importance.importances)

        importances = np.reshape(np.stack(importances, axis=2), (X.shape[1], -1), 'F')
        bunch = sk_utils.Bunch(importances_mean=np.mean(importances, axis=1),
                               importances_std=np.std(importances, axis=1),
                               importances=importances,
                               columns=X.columns.to_list())
        return bunch

    @staticmethod
    def detect_strategy(strategy, *, threshold=None, quantile=None, number=None,
                        default_strategy, default_threshold, default_quantile, default_number):
        if strategy is None:
            if threshold is not None:
                strategy = ToolBox.STRATEGY_THRESHOLD
            elif number is not None:
                strategy = ToolBox.STRATEGY_NUMBER
            elif quantile is not None:
                strategy = ToolBox.STRATEGY_QUANTILE
            else:
                strategy = default_strategy

        if strategy == ToolBox.STRATEGY_THRESHOLD:
            if threshold is None:
                threshold = default_threshold
        elif strategy == ToolBox.STRATEGY_NUMBER:
            if number is None:
                number = default_number
        elif strategy == ToolBox.STRATEGY_QUANTILE:
            if quantile is None:
                quantile = default_quantile
            assert 0 < quantile < 1.0
        else:
            raise ValueError(f'Unsupported strategy: {strategy}')

        return strategy, threshold, quantile, number

    _default_strategy_of_feature_importance_selection = dict(
        default_strategy=STRATEGY_THRESHOLD,
        default_threshold=0.1,
        default_quantile=0.2,
        default_number=0.8,
    )

    @classmethod
    def detect_strategy_of_feature_selection_by_importance(cls, strategy, *, threshold=None, quantile=None,
                                                           number=None):
        return cls.detect_strategy(strategy, threshold=threshold, number=number, quantile=quantile,
                                   **cls._default_strategy_of_feature_importance_selection)

    @classmethod
    def select_feature_by_importance(cls, feature_importance,
                                     strategy=None, threshold=None, quantile=None, number=None):
        assert isinstance(feature_importance, (list, tuple, np.ndarray)) and len(feature_importance) > 0

        strategy, threshold, quantile, number = cls.detect_strategy_of_feature_selection_by_importance(
            strategy, threshold=threshold, quantile=quantile, number=number)

        feature_importance = np.array(feature_importance)
        idx = np.arange(len(feature_importance))

        if strategy == ToolBox.STRATEGY_THRESHOLD:
            selected = np.where(np.where(feature_importance >= threshold, idx, -1) >= 0)[0]
        elif strategy == ToolBox.STRATEGY_QUANTILE:
            q = np.quantile(feature_importance, quantile)
            selected = np.where(np.where(feature_importance >= q, idx, -1) >= 0)[0]
        elif strategy == ToolBox.STRATEGY_NUMBER:
            if isinstance(number, float) and 0 < number < 1.0:
                number = math.ceil(len(feature_importance) * number)
            pos = len(feature_importance) - number
            sorted_ = np.argsort(np.argsort(feature_importance))
            selected = np.where(sorted_ >= pos)[0]
        else:
            raise ValueError(f'Unsupported strategy: {strategy}')

        unselected = list(set(range(len(feature_importance))) - set(selected))
        unselected = np.array(unselected)

        return selected, unselected

    # reused utilities
    train_test_split = sk_ms.train_test_split
    compute_class_weight = sk_utils.compute_class_weight

    @staticmethod
    def compute_sample_weight(y):
        return sk_utils.compute_sample_weight('balanced', y)

    # reused modules
    # drift_detection = drift_detection_
    # feature_importance = feature_importance_
    # feature_generators = feature_generators_
    column_selector = column_selector_
    metrics = metrics_.Metrics

    _data_hasher_cls = data_hasher_.DataHasher
    _data_cleaner_cls = data_cleaner_.DataCleaner
    _estimator_detector_cls = estimator_detector_.EstimatorDetector
    _collinearity_detector_cls = collinearity_.MultiCollinearityDetector
    _drift_detector_cls = drift_detection_.DriftDetector
    _feature_selector_with_drift_detection_cls = drift_detection_.FeatureSelectorWithDriftDetection
    _pseudo_labeling_cls = pseudo_labeling_.PseudoLabeling
    _kfold_cls = sk_ms.KFold
    _stratified_kfold_cls = sk_ms.StratifiedKFold
    _greedy_ensemble_cls = ensemble_.GreedyEnsemble

    @classmethod
    def data_hasher(cls, method='md5'):
        return cls._data_hasher_cls(method=method)

    @classmethod
    def data_cleaner(cls, nan_chars=None, correct_object_dtype=True, drop_constant_columns=True,
                     drop_duplicated_columns=False, drop_label_nan_rows=True, drop_idness_columns=True,
                     replace_inf_values=np.nan, drop_columns=None, reserve_columns=None,
                     reduce_mem_usage=False, int_convert_to='float'):
        return cls._data_cleaner_cls(
            nan_chars=nan_chars, correct_object_dtype=correct_object_dtype,
            drop_constant_columns=drop_constant_columns, drop_duplicated_columns=drop_duplicated_columns,
            drop_label_nan_rows=drop_label_nan_rows, drop_idness_columns=drop_idness_columns,
            replace_inf_values=replace_inf_values, drop_columns=drop_columns,
            reserve_columns=reserve_columns, reduce_mem_usage=reduce_mem_usage,
            int_convert_to=int_convert_to)

    @classmethod
    def estimator_detector(cls, name_or_cls, task, *,
                           init_kwargs=None, fit_kwargs=None, n_samples=100, n_features=5):
        return cls._estimator_detector_cls(
            name_or_cls, task,
            init_kwargs=init_kwargs, fit_kwargs=fit_kwargs, n_samples=n_samples, n_features=n_features)

    @classmethod
    def collinearity_detector(cls):
        return cls._collinearity_detector_cls()

    @classmethod
    def drift_detector(cls, preprocessor=None, estimator=None, random_state=None):
        return cls._drift_detector_cls(preprocessor=preprocessor, estimator=estimator, random_state=random_state)

    @classmethod
    def feature_selector_with_drift_detection(cls, remove_shift_variable=True, variable_shift_threshold=0.7,
                                              variable_shift_scorer=None,
                                              auc_threshold=0.55, min_features=10, remove_size=0.1,
                                              sample_balance=True, max_test_samples=None, cv=5, random_state=None,
                                              callbacks=None):
        return cls._feature_selector_with_drift_detection_cls(
            remove_shift_variable=remove_shift_variable, variable_shift_threshold=variable_shift_threshold,
            variable_shift_scorer=variable_shift_scorer,
            auc_threshold=auc_threshold, min_features=min_features, remove_size=remove_size,
            sample_balance=sample_balance, max_test_samples=max_test_samples, cv=cv, random_state=random_state,
            callbacks=callbacks)

    @classmethod
    def feature_selector_with_feature_importances(cls, strategy=None, threshold=None, quantile=None, number=None):
        pass

    @classmethod
    def pseudo_labeling(cls, strategy, threshold=None, quantile=None, number=None):
        return cls._pseudo_labeling_cls(strategy, threshold=threshold, quantile=quantile, number=number)

    @classmethod
    def kfold(cls, n_splits=5, *, shuffle=False, random_state=None):
        return cls._kfold_cls(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    @classmethod
    def statified_kfold(cls, n_splits=5, *, shuffle=False, random_state=None):
        return cls._stratified_kfold_cls(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    @classmethod
    def greedy_ensemble(cls, task, estimators, need_fit=False, n_folds=5, method='soft', random_state=9527,
                        scoring='neg_log_loss', ensemble_size=0):
        return cls._greedy_ensemble_cls(task, estimators, need_fit=need_fit, n_folds=n_folds, method=method,
                                        random_state=random_state, scoring=scoring, ensemble_size=ensemble_size)


_predefined_transformers = dict(
    Pipeline=pipeline.Pipeline,
    # SimpleImputer=sk_imp.SimpleImputer,  # replaced with sklearn_ex SafeSimpleImputer
    StandardScaler=sk_pre.StandardScaler,
    MinMaxScaler=sk_pre.MinMaxScaler,
    MaxAbsScaler=sk_pre.MaxAbsScaler,
    RobustScaler=sk_pre.RobustScaler,
    Normalizer=sk_pre.Normalizer,
    KBinsDiscretizer=sk_pre.KBinsDiscretizer,
    LabelEncoder=sk_pre.LabelEncoder,
    OrdinalEncoder=sk_pre.OrdinalEncoder,
    OneHotEncoder=sk_pre.OneHotEncoder,
    PolynomialFeatures=sk_pre.PolynomialFeatures,
    QuantileTransformer=sk_pre.QuantileTransformer,
    PowerTransformer=sk_pre.PowerTransformer,
    PCA=sk_dec.PCA,
    TruncatedSVD=sk_dec.TruncatedSVD,
    DataFrameMapper=dataframe_mapper_.DataFrameMapper,

    # PassThroughEstimator=sk_ex.PassThroughEstimator,
    # MultiLabelEncoder=sk_ex.MultiLabelEncoder,
    # SafeOrdinalEncoder=sk_ex.SafeOrdinalEncoder,
    # SafeOneHotEncoder=sk_ex.SafeOneHotEncoder,
    # AsTypeTransformer=sk_ex.AsTypeTransformer,
    # SafeLabelEncoder=sk_ex.SafeLabelEncoder,
    # LogStandardScaler=sk_ex.LogStandardScaler,
    # SkewnessKurtosisTransformer=sk_ex.SkewnessKurtosisTransformer,
    # FeatureSelectionTransformer=sk_ex.FeatureSelectionTransformer,
    # FloatOutputImputer=sk_ex.FloatOutputImputer,
    # LgbmLeavesEncoder=sk_ex.LgbmLeavesEncoder,
    # CategorizeEncoder=sk_ex.CategorizeEncoder,
    # MultiKBinsDiscretizer=sk_ex.MultiKBinsDiscretizer,
    # DataFrameWrapper=sk_ex.DataFrameWrapper,
    # GaussRankScaler=sk_ex.GaussRankScaler,
    # VarLenFeatureEncoder=sk_ex.VarLenFeatureEncoder,
    # MultiVarLenFeatureEncoder=sk_ex.MultiVarLenFeatureEncoder,
    # LocalizedTfidfVectorizer=sk_ex.LocalizedTfidfVectorizer,
    # TfidfEncoder=sk_ex.TfidfEncoder,
    # DatetimeEncoder=sk_ex.DatetimeEncoder,

    # FeatureGenerationTransformer=feature_generators_.FeatureGenerationTransformer,
)

if feature_generators_.is_feature_generator_ready:
    _predefined_transformers['FeatureGenerationTransformer'] = feature_generators_.FeatureGenerationTransformer

for name, tf in _predefined_transformers.items():
    register_transformer(tf, name=name, dtypes=pd.DataFrame)

__all__ = [
    ToolBox.__name__,
]
