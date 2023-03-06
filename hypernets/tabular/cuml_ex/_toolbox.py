# -*- coding:utf-8 -*-
"""

"""
import copy
import inspect
from functools import partial

import cudf
import cupy
import numpy as np
import pandas as pd
from cuml.common.array import CumlArray

from hypernets.utils import const, logging
from . import _data_cleaner
from . import _dataframe_mapper, _metrics, _data_hasher, _model_selection, _ensemble, _drift_detection
from . import _estimator_detector
from . import _pseudo_labeling
from . import _transformer  # NOQA,  register customized transformer
from .. import sklearn_ex as sk_ex
from ..toolbox import ToolBox, register_transformer, randint

try:
    import xgboost

    is_xgboost_installed = True
except ImportError:
    is_xgboost_installed = False

logger = logging.get_logger(__name__)

if not cupy.cuda.is_available():
    logger.warning('Not found available CUDA devices.')


class CumlToolBox(ToolBox):
    acceptable_types = (cupy.ndarray, cudf.DataFrame, cudf.Series)

    is_available = cupy.cuda.is_available()

    @staticmethod
    def is_cudf_dataframe(x):
        return isinstance(x, cudf.DataFrame)

    @staticmethod
    def is_cudf_series(x):
        return isinstance(x, cudf.Series)

    @staticmethod
    def is_cudf_dataframe_or_series(x):
        return isinstance(x, (cudf.DataFrame, cudf.Series))

    @staticmethod
    def is_cupy_array(x):
        return isinstance(x, cupy.ndarray)

    @staticmethod
    def is_cuml_object(x):
        return isinstance(x, (cudf.DataFrame, cudf.Series, cupy.ndarray, CumlArray))

    @staticmethod
    def exist_cudf_dataframe_or_series(*args):
        return any(map(CumlToolBox.is_cudf_dataframe_or_series, args))

    @staticmethod
    def exist_cupy_array(*args):
        return any(map(CumlToolBox.is_cupy_array, args))

    @staticmethod
    def exist_cuml_object(*args):
        return any(map(CumlToolBox.is_cuml_object, args))

    @staticmethod
    def memory_total():
        device = cupy.cuda.Device(0)
        return device.mem_info[1]

    @staticmethod
    def memory_free():
        device = cupy.cuda.Device(0)
        return device.mem_info[0]

    @staticmethod
    def memory_usage(*data):
        usage = 0
        for x in data:
            if isinstance(x, cudf.DataFrame):
                usage += x.memory_usage().sum()
            elif isinstance(x, cudf.Series):
                usage += x.memory_usage()
            elif isinstance(x, cupy.ndarray):
                usage += x.nbytes
            else:
                pass  # ignore
        return usage

    @staticmethod
    def to_local(*data):
        def to_np_or_pd(x):
            if isinstance(x, (cudf.DataFrame, cudf.Series)) or hasattr(x, 'to_pandas'):
                return x.to_pandas()
            elif isinstance(x, cupy.ndarray):
                return cupy.asnumpy(x)
            elif isinstance(x, CumlArray):
                return cupy.asnumpy(cupy.asarray(x))
            elif isinstance(x, list):
                return [to_np_or_pd(i) for i in x]
            elif isinstance(x, tuple):
                return tuple(to_np_or_pd(i) for i in x)
            elif isinstance(x, dict):
                return {k: to_np_or_pd(v) for k, v in x.items()}
            else:
                return x

        return [to_np_or_pd(x) for x in data]

    @staticmethod
    def from_local(*data, enable_cuml_array=False):
        def from_np_or_pd(x):
            if isinstance(x, (pd.DataFrame, pd.Series)):
                return cudf.from_pandas(x)
            elif isinstance(x, np.ndarray):
                # print('-' * 10, x, x.dtype)
                # return cupy.array(x)
                sdtype = str(x.dtype)
                if sdtype.find('int') >= 0 or sdtype.find('float') >= 0:  # cupy does not support object
                    if enable_cuml_array:
                        return CumlArray(x)
                    else:
                        return cupy.array(x)
                elif x.ndim > 1 and x.shape[1] > 1:
                    return cudf.DataFrame(x)
                else:
                    return cudf.Series(x)
            elif isinstance(x, (list, tuple)):
                return [from_np_or_pd(i) for i in x]
            elif isinstance(x, dict):
                return {k: from_np_or_pd(v) for k, v in x.items()}
            else:
                return x

        return [from_np_or_pd(x) for x in data]

    @staticmethod
    def load_data(data_path, *, reset_index=False, reader_mapping=None, **kwargs):
        if reader_mapping is None:
            reader_mapping = {
                'csv': cudf.read_csv,
                'txt': cudf.read_csv,
                'parquet': cudf.read_parquet,
                'par': cudf.read_parquet,
                'json': cudf.read_json,
            }
        df = ToolBox.load_data(data_path, reset_index=reset_index, reader_mapping=reader_mapping, **kwargs)
        return df

    @staticmethod
    def parquet():
        from . import _persistence
        return _persistence.CumlParquetPersistence()

    @staticmethod
    def unique(y):
        if isinstance(y, cudf.Series):
            uniques = y.unique().to_pandas().values
            uniques = set(uniques)
        elif isinstance(y, cupy.ndarray):
            uniques = cudf.Series(y).unique().to_pandas().values
            uniques = set(uniques)
        else:
            uniques = ToolBox.unique(y)

        return uniques

    @staticmethod
    def nunique_df(df):
        if isinstance(df, cudf.DataFrame):
            columns = df.columns.to_list()
            uniques = [df[c].nunique() for c in columns]
            return {c: v for c, v in zip(columns, uniques)}
        else:
            return ToolBox.nunique_df(df)

    @staticmethod
    def value_counts(ar):
        return cudf.Series(ar).value_counts().to_pandas().to_dict()

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

        fn = cupy.vectorize(_collapse, otypes=[object], signature='(m)->()')
        t = fn(arr)
        if keep_dim:
            shape = arr.shape[:-1] + (1,)
            t = t.reshape(shape)
        return t

    # @staticmethod
    # def reset_index(X):
    #     return ToolBox.reset_index(X)

    # @staticmethod
    # def select_df(df, indices):
    #     return ToolBox.select_df(df, indices=indices)

    @staticmethod
    def stack_array(arrs, axis=0):
        assert axis in (0, 1)

        if not CumlToolBox.exist_cuml_object(*arrs):
            return ToolBox.stack_array(arrs, axis=axis)

        arrs = CumlToolBox.from_local(*arrs)

        if all(map(CumlToolBox.is_cudf_dataframe_or_series, arrs)):
            return cudf.concat(arrs, axis=axis, ignore_index=True)

        ndims = set([len(a.shape) for a in arrs])
        if len(ndims) > 1:
            assert ndims == {1, 2}
            assert all([len(a.shape) == 1 or a.shape[1] == 1 for a in arrs])
            arrs = [a.reshape(-1, 1) if len(a.shape) == 1 else a for a in arrs]
        axis = min(axis, min([len(a.shape) for a in arrs]) - 1)
        assert axis >= 0

        return cupy.concatenate(arrs, axis=axis)

    @staticmethod
    def take_array(arr, indices, axis=None):
        assert len(arr) > 0

        if CumlToolBox.is_cudf_series(arr):
            # return arr.take(indices, keep_index=False)
            return arr.take(indices).reset_index(drop=True)

        atype = type(arr[0]).__name__
        if atype.find('int') >= 0 or atype.find('float') >= 0:
            return cupy.take(cupy.array(arr), indices=cupy.array(indices), axis=axis)
        else:
            arr = cudf.from_pandas(pd.Series(arr))
            # return arr.take(indices, keep_index=False)
            return arr.take(indices).reset_index(drop=True)

    @staticmethod
    def array_to_df(arr, *, columns=None, index=None, meta=None):
        return cudf.DataFrame(arr, columns=columns, index=index)

    @staticmethod
    def merge_oof(oofs):
        row_count = sum(map(lambda x: len(x[0]), oofs))
        max_idx = max(map(lambda x: np.max(x[0]), oofs))
        if max_idx >= row_count:
            row_count = max_idx + 1

        proba = oofs[0][1]
        if len(proba.shape) == 1:
            r = cupy.full(row_count, np.nan, proba.dtype)
        else:
            r = cupy.full((row_count, proba.shape[-1]), np.nan, proba.dtype)

        for idx, proba in oofs:
            r[idx] = proba

        return r

    merge_oof.__doc__ = ToolBox.merge_oof.__doc__

    @staticmethod
    def select_valid_oof(y, oof):
        if isinstance(oof, cupy.ndarray):
            if len(oof.shape) == 1:
                idx = cupy.argwhere(~cupy.isnan(oof[:])).ravel()
            elif len(oof.shape) == 2:
                idx = cupy.argwhere(~cupy.isnan(oof[:, 0])).ravel()
            elif len(oof.shape) == 3:
                idx = cupy.argwhere(~cupy.isnan(oof[:, 0, 0])).ravel()
            else:
                raise ValueError(f'Unsupported shape:{oof.shape}')
            return y.iloc[idx] if hasattr(y, 'iloc') else y[idx], oof[idx]
        else:
            return ToolBox.select_valid_oof(y, oof)

    @staticmethod
    def mean_oof(probas):
        if any(map(CumlToolBox.is_cuml_object, probas)):
            probas = cupy.array([cupy.array(p) for p in probas])
            proba = cupy.mean(probas, axis=0)
        else:
            proba = ToolBox.mean_oof(probas)
        return proba

    @staticmethod
    def concat_df(dfs, axis=0, repartition=False, random_state=9527, **kwargs):
        if not any(map(CumlToolBox.is_cuml_object, dfs)):
            return ToolBox.concat_df(dfs, axis=axis, repartition=repartition, random_state=random_state, **kwargs)

        header = dfs[0]
        assert isinstance(header, (cudf.DataFrame, cudf.Series))

        def to_cudf_type(t):
            if isinstance(t, (cudf.DataFrame, cudf.Series)):
                return t
            else:
                if not CumlToolBox.is_cuml_object(t):
                    t, = CumlToolBox.from_local(t)
                if isinstance(header, cudf.Series):
                    return cudf.Series(t, name=header.name)
                else:
                    return cudf.DataFrame(t, columns=header.columns)

        dfs = [header] + [to_cudf_type(df) for df in dfs[1:]]

        df = cudf.concat(dfs, axis=axis, **kwargs)
        if repartition:
            df = df.sample(frac=1.0, random_state=random_state)
        return df

    @staticmethod
    def fix_binary_predict_proba_result(proba):
        if proba.ndim == 1:
            if CumlToolBox.is_cupy_array(proba):
                proba = cupy.vstack([1 - proba, proba]).T
            else:
                proba = cudf.Series(proba)
                proba = cudf.concat([1 - proba, proba], axis=1)
        elif proba.shape[1] == 1:
            proba = cupy.hstack([1 - proba, proba])

        return proba

    @staticmethod
    def permutation_importance(estimator, X, y, *args, scoring=None, n_repeats=5,
                               n_jobs=None, random_state=None):
        # if not CumlToolBox.is_cuml_object(X):
        #     return ToolBox.permutation_importance(estimator, X, y, *args,
        #                                           scoring=scoring,
        #                                           n_repeats=n_repeats,
        #                                           n_jobs=n_jobs,
        #                                           random_state=random_state)
        # from sklearn import utils as sk_utils
        # from sklearn.metrics import check_scoring
        # scorer = check_scoring(CumlToolBox.wrap_for_local_scorer(estimator, CumlToolBox.infer_task_type(y)), scoring)
        # y, = CumlToolBox.to_local(y)
        # baseline_score = scorer(estimator, X, y)
        # scores = []
        #
        # X = X.reset_index(drop=True)
        # for c in X.columns:
        #     col_scores = []
        #     X_permuted = X.copy()
        #     for i in range(n_repeats):
        #         rnd = random_state.randint(1, 65535) if hasattr(random_state, 'randint') else randint()
        #         X_permuted[c] = X[c].sample(n=X.shape[0], random_state=rnd).reset_index(drop=True)
        #         col_scores.append(scorer(estimator, X_permuted, y))
        #     if logger.is_debug_enabled():
        #         logger.debug(f'permuted scores [{c}]: {col_scores}')
        #     scores.append(col_scores)
        #
        # importances = baseline_score - np.array(scores)
        # return sk_utils.Bunch(importances_mean=np.mean(importances, axis=1),
        #                       importances_std=np.std(importances, axis=1),
        #                       importances=importances)

        estimator = CumlToolBox.wrap_for_local_scorer(estimator, CumlToolBox.infer_task_type(y))
        X, y = CumlToolBox.to_local(X, y)
        return ToolBox.permutation_importance(estimator, X, y, *args,
                                              scoring=scoring,
                                              n_repeats=n_repeats,
                                              n_jobs=n_jobs,
                                              random_state=random_state)

    @staticmethod
    def compute_class_weight(class_weight, *, classes, y):
        assert isinstance(y, (cupy.ndarray, cudf.Series))

        if CumlToolBox.is_cupy_array(y):
            y = cudf.Series(y)

        if class_weight == 'balanced':
            # Find the weight of each class as present in y.
            # le = tfs.LabelEncoder()
            # y_ind = le.fit_transform(y)
            # if not all(np.in1d(classes, le.classes_)):
            #     raise ValueError("classes should have valid labels that are in y")
            # recip_freq = len(y) / (len(le.classes_) *
            #                        np.bincount(y_ind).astype(np.float64))
            # weight = recip_freq[le.transform(classes)]

            # if not all(np.in1d(np.array(classes), le.classes_.to_array())):
            #     raise ValueError("classes should have valid labels that are in y")
            #
            # recip_freq = len(y) / (len(le.classes_) *
            #                        cupy.bincount(y_ind.values).astype(cupy.float64))
            # ct = le.transform(cudf.from_pandas(pd.Series(classes)))
            # weight = recip_freq[ct.values].tolist()

            value_counts = y.value_counts().astype(cupy.float64)
            if not (value_counts.index.isin(classes).all() and len(value_counts) == len(classes)):
                raise ValueError("classes should have valid labels that are in y")
            recip_freq = len(y) / (len(classes) * value_counts)
            weight = recip_freq.reindex(classes).to_pandas().values
        else:
            raise ValueError("Only class_weight == 'balanced' is supported.")

        return weight

    @staticmethod
    def compute_sample_weight(y):
        assert len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1)

        if isinstance(y, cupy.ndarray):
            y = cudf.Series(y)
        if isinstance(y, (cupy.ndarray, cudf.Series)):
            classes = cudf.Series(y).unique().to_pandas().values
        else:
            classes = np.unique(y)
        classes_weights = list(CumlToolBox.compute_class_weight('balanced', classes=classes, y=y))

        sample_weight = cupy.ones(y.shape[0])
        for i, c in enumerate(classes):
            indices = (y == c)
            if isinstance(indices, (pd.Series, cudf.Series)):
                indices = indices.values
            sample_weight[indices] *= classes_weights[i]

        return sample_weight

    @staticmethod
    def call_local(fn, *args, input_to_local=None, output_to_cuml=None, **kwargs):
        def _exist_cu_type(*args_):
            for x in args_:
                found = False
                if isinstance(x, CumlToolBox.acceptable_types):
                    found = True
                elif isinstance(x, (list, tuple)):
                    found = _exist_cu_type(*x)
                elif isinstance(x, dict):
                    found = _exist_cu_type(*x.values())
                if found:
                    return found

            return False

        found_cu_type = _exist_cu_type(*args) or _exist_cu_type(*kwargs.values())
        if found_cu_type and input_to_local is not False:
            args = CumlToolBox.to_local(*args)
            kwargs = CumlToolBox.to_local(kwargs)[0]

        r = fn(*args, **kwargs)
        if found_cu_type and r is not None and output_to_cuml is not False:
            r = CumlToolBox.from_local(r)[0]

        return r

    @staticmethod
    def wrap_for_local_scorer(estimator, target_type):
        def _call_with_local(fn_call, fn_fix, *args, **kwargs):
            params = inspect.signature(fn_call).parameters.keys()
            if 'to_local' in params:
                kwargs['to_local'] = True

            r = fn_call(*args, **kwargs)
            if CumlToolBox.is_cuml_object(r):
                r, = CumlToolBox.to_local(r)
                if callable(fn_fix):
                    r = fn_fix(r)
            return r

        if hasattr(estimator, 'as_local'):
            estimator = estimator.as_local()
            return estimator

        estimator = copy.copy(estimator)
        if hasattr(estimator, 'predict_proba'):
            orig_predict_proba = estimator.predict_proba
            fix = CumlToolBox.fix_binary_predict_proba_result if target_type == const.TASK_BINARY else None
            setattr(estimator, '_orig_predict_proba', orig_predict_proba)
            setattr(estimator, 'predict_proba', partial(_call_with_local, orig_predict_proba, fix))

        if hasattr(estimator, 'predict'):
            orig_predict = estimator.predict
            setattr(estimator, '_orig_predict', orig_predict)
            setattr(estimator, 'predict', partial(_call_with_local, orig_predict, None))

        return estimator

    @staticmethod
    def wrap_local_estimator(estimator, *, methods=None):
        if methods is None:
            methods = ('fit', 'fit_cross_validation', 'predict', 'predict_proba')

        for fn_name in methods:
            fn_name_original = f'_wrapped_{fn_name}_by_wle'
            if hasattr(estimator, fn_name) and not hasattr(estimator, fn_name_original):
                fn = getattr(estimator, fn_name)
                assert callable(fn)
                setattr(estimator, fn_name_original, fn)
                setattr(estimator, fn_name, partial(CumlToolBox.call_local, fn))
                # print('wrapped:', fn_name)
        return estimator

    @staticmethod
    def unwrap_local_estimator(estimator, *, methods=None, copy=False):
        if methods is None:
            methods = ('fit', 'fit_cross_validation', 'predict', 'predict_proba')

        for fn_name in methods:
            fn_name_original = f'_wrapped_{fn_name}_by_wle'
            if hasattr(estimator, fn_name) and hasattr(estimator, fn_name_original):
                fn = getattr(estimator, fn_name_original)
                assert callable(fn)
                setattr(estimator, fn_name, fn)
                delattr(estimator, fn_name_original)
        return estimator

    @classmethod
    def general_preprocessor(cls, X, y=None):
        if isinstance(X, CumlToolBox.acceptable_types):
            cs = cls.column_selector
            tfs = cls.transformers

            cat_steps = [('imputer_cat', tfs['ConstantImputer'](fill_value='')),
                         # ('encoder', tfs['SafeOrdinalEncoder']()),
                         ('encoder', tfs['MultiLabelEncoder']()),
                         ]
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
        else:
            return ToolBox.general_preprocessor(X, y)

    @classmethod
    def general_estimator(cls, X, y=None, estimator=None, task=None):
        def default_xgb(task_):
            from ._estimator import AdaptedXGBClassifier, AdaptedXGBRegressor
            options = dict(n_estimators=100,
                           tree_method='gpu_hist',
                           # num_leaves=15,
                           max_depth=5,
                           min_child_weight=5,
                           learning_rate=0.1,
                           gamma=1,
                           reg_alpha=1,
                           reg_lambda=1,
                           random_state=randint())
            if task_ == const.TASK_REGRESSION:
                est_cls = AdaptedXGBRegressor
            else:
                options['use_label_encoder'] = False
                est_cls = AdaptedXGBClassifier
            return est_cls(**options)

        def default_rf(task_):
            from ._estimator import AdaptedRandomForestClassifier, AdaptedRandomForestRegressor
            est_cls = AdaptedRandomForestRegressor if task_ == const.TASK_REGRESSION else AdaptedRandomForestClassifier
            return est_cls(min_samples_leaf=20, min_impurity_decrease=0.01, random_state=randint())

        if estimator is None:
            estimator = 'xgb' if is_xgboost_installed else 'rf'
        if task is None:
            assert y is not None, '"y" or "task" is required.'
            task = cls.infer_task_type(y)

        if estimator == 'xgb':
            estimator_ = default_xgb(task)
        elif estimator == 'rf':
            estimator_ = default_rf(task)
        else:
            estimator_ = ToolBox.general_estimator(X, y=y, estimator=estimator, task=task)
            estimator_ = cls.wrap_local_estimator(estimator_)

        return estimator_

    train_test_split = _model_selection.train_test_split
    metrics = _metrics.CumlMetrics
    _data_hasher_cls = _data_hasher.CumlDataHasher
    _data_cleaner_cls = _data_cleaner.CumlDataCleaner
    _estimator_detector_cls = _estimator_detector.CumlEstimatorDetector
    _kfold_cls = _model_selection.FakeKFold
    _stratified_kfold_cls = _model_selection.FakeStratifiedKFold
    _feature_selector_with_drift_detection_cls = _drift_detection.CumlFeatureSelectorWithDriftDetection
    _pseudo_labeling_cls = _pseudo_labeling.CumlPseudoLabeling
    _greedy_ensemble_cls = _ensemble.CumlGreedyEnsemble

    @classmethod
    def collinearity_detector(cls):
        raise NotImplementedError('collinearity_detector is not implemented with cuml.')


_predefined_transformers = dict(
    PassThroughEstimator=sk_ex.PassThroughEstimator,
    AsTypeTransformer=sk_ex.AsTypeTransformer,

    # Pipeline=tfs.LocalizablePipeline,
    # SimpleImputer=tfs.LocalizableSimpleImputer,
    # ConstantImputer=tfs.ConstantImputer,
    # StandardScaler=tfs.LocalizableStandardScaler,
    # MinMaxScaler=tfs.LocalizableMinMaxScaler,
    # MaxAbsScaler=tfs.LocalizableMaxAbsScaler,
    # RobustScaler=tfs.LocalizableRobustScaler,
    # # Normalizer=tfs.Normalizer,
    # # KBinsDiscretizer=tfs.KBinsDiscretizer,
    # LabelEncoder=tfs.LocalizableLabelEncoder,
    # # OrdinalEncoder=tfs.OrdinalEncoder,
    # OneHotEncoder=tfs.LocalizableOneHotEncoder,
    # TargetEncoder=tfs.TargetEncoder,
    # # PolynomialFeatures=sk_pre.PolynomialFeatures,
    # # QuantileTransformer=sk_pre.QuantileTransformer,
    # # PowerTransformer=sk_pre.PowerTransformer,
    # # PCA=sk_dec.PCA,
    # TruncatedSVD=tfs.LocalizableTruncatedSVD,
    DataFrameMapper=_dataframe_mapper.CumlDataFrameMapper,
    # MultiLabelEncoder=tfs.MultiLabelEncoder,
    # # SafeOrdinalEncoder=tfs.SafeOrdinalEncoder,
    # # SafeOneHotEncoder=sk_ex.SafeOneHotEncoder,
    # SafeLabelEncoder=tfs.SafeLabelEncoder,
    # # LogStandardScaler=tfs.LogStandardScaler,
    # # SkewnessKurtosisTransformer=sk_ex.SkewnessKurtosisTransformer,
    # # FeatureSelectionTransformer=sk_ex.FeatureSelectionTransformer,
    # # FloatOutputImputer=tfs.FloatOutputImputer,
    # # LgbmLeavesEncoder=sk_ex.LgbmLeavesEncoder,
    # # CategorizeEncoder=tfs.CategorizeEncoder,
    # # MultiKBinsDiscretizer=sk_ex.MultiKBinsDiscretizer,
    # # DataFrameWrapper=tfs.DataFrameWrapper,
    # # GaussRankScaler=sk_ex.GaussRankScaler,
    # # VarLenFeatureEncoder=sk_ex.VarLenFeatureEncoder,
    # # MultiVarLenFeatureEncoder=sk_ex.MultiVarLenFeatureEncoder,
    # # LocalizedTfidfVectorizer=sk_ex.LocalizedTfidfVectorizer,
    # # TfidfEncoder=sk_ex.TfidfEncoder,
    # # DatetimeEncoder=sk_ex.DatetimeEncoder,
    # # FeatureGenerationTransformer=feature_generators_.FeatureGenerationTransformer,
    FeatureImportancesSelectionTransformer=sk_ex.FeatureImportancesSelectionTransformer,
)
for name, tf in _predefined_transformers.items():
    register_transformer(tf, name=name, dtypes=cudf.DataFrame)
