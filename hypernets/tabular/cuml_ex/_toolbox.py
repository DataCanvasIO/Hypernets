# -*- coding:utf-8 -*-
"""

"""
from functools import partial

import cudf
import cupy
import numpy as np
import pandas as pd
from cuml.common.array import CumlArray

from . import _dataframe_mapper, _transformer as tfs, _metrics, _data_hasher, _model_selection, _ensemble
from ..toolbox import ToolBox


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
    def from_local(*data):
        def from_np_or_pd(x):
            if isinstance(x, (pd.DataFrame, pd.Series)):
                return cudf.from_pandas(x)
            elif isinstance(x, np.ndarray):
                # print('-' * 10, x, x.dtype)
                # return cupy.array(x)
                sdtype = str(x.dtype)
                if sdtype.find('int') >= 0 or sdtype.find('float') >= 0:  # cupy does not support object
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
    def value_counts(ar):
        return cudf.Series(ar).value_counts().to_pandas().to_dict()

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
        if all(map(CumlToolBox.is_cudf_series, [arr, indices])):
            return arr.take(indices, keep_index=False)

        return cupy.take(cupy.array(arr), indices=cupy.array(indices), axis=axis)

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
    def concat_df(dfs, axis=0, repartition=False, **kwargs):
        return cudf.concat(dfs, axis=axis, **kwargs)

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
        raise NotImplementedError()  # fixme

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
            sample_weight[y == c] *= classes_weights[i]

        return sample_weight

    @staticmethod
    def call_local(fn, *args, **kwargs):
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
        if found_cu_type:
            args = CumlToolBox.to_local(*args)
            kwargs = CumlToolBox.to_local(kwargs)[0]

        r = fn(*args, **kwargs)
        if found_cu_type and r is not None:
            r = CumlToolBox.from_local(r)[0]

        return r

    @staticmethod
    def wrap_local_estimator(estimator):
        for fn_name in ('fit', 'fit_cross_validation', 'predict', 'predict_proba'):
            fn_name_original = f'_wrapped_{fn_name}_by_wle'
            if hasattr(estimator, fn_name) and not hasattr(estimator, fn_name_original):
                fn = getattr(estimator, fn_name)
                assert callable(fn)
                setattr(estimator, fn_name_original, fn)
                setattr(estimator, fn_name, partial(CumlToolBox.call_local, fn))
                # print('wrapped:', fn_name)
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
        est = ToolBox.general_estimator(X, y, estimator=estimator, task=task)

        return cls.wrap_local_estimator(est)

    train_test_split = _model_selection.train_test_split
    metrics = _metrics.CumlMetrics
    _data_hasher_cls = _data_hasher.CumlDataHasher

    _kfold_cls = _model_selection.FakeKFold
    _stratified_kfold_cls = _model_selection.FakeStratifiedKFold

    _greedy_ensemble_cls = _ensemble.CumlGreedyEnsemble

    transformers = dict(
        Pipeline=tfs.LocalizablePipeline,
        SimpleImputer=tfs.LocalizableSimpleImputer,
        ConstantImputer=tfs.ConstantImputer,
        StandardScaler=tfs.LocalizableStandardScaler,
        MinMaxScaler=tfs.LocalizableMinMaxScaler,
        MaxAbsScaler=tfs.LocalizableMaxAbsScaler,
        RobustScaler=tfs.LocalizableRobustScaler,
        # Normalizer=tfs.Normalizer,
        # KBinsDiscretizer=tfs.KBinsDiscretizer,
        LabelEncoder=tfs.LocalizableLabelEncoder,
        # OrdinalEncoder=tfs.OrdinalEncoder,
        OneHotEncoder=tfs.LocalizableOneHotEncoder,
        TargetEncoder=tfs.TargetEncoder,
        # PolynomialFeatures=sk_pre.PolynomialFeatures,
        # QuantileTransformer=sk_pre.QuantileTransformer,
        # PowerTransformer=sk_pre.PowerTransformer,
        # PCA=sk_dec.PCA,
        TruncatedSVD=tfs.LocalizableTruncatedSVD,
        DataFrameMapper=_dataframe_mapper.CumlDataFrameMapper,
        PassThroughEstimator=tfs.PassThroughEstimator,
        MultiLabelEncoder=tfs.MultiLabelEncoder,
        # SafeOrdinalEncoder=tfs.SafeOrdinalEncoder,
        # SafeOneHotEncoder=sk_ex.SafeOneHotEncoder,
        AsTypeTransformer=tfs.AsTypeTransformer,
        SafeLabelEncoder=tfs.SafeLabelEncoder,
        # LogStandardScaler=tfs.LogStandardScaler,
        # SkewnessKurtosisTransformer=sk_ex.SkewnessKurtosisTransformer,
        # FeatureSelectionTransformer=sk_ex.FeatureSelectionTransformer,
        # FloatOutputImputer=tfs.FloatOutputImputer,
        # LgbmLeavesEncoder=sk_ex.LgbmLeavesEncoder,
        # CategorizeEncoder=tfs.CategorizeEncoder,
        # MultiKBinsDiscretizer=sk_ex.MultiKBinsDiscretizer,
        # DataFrameWrapper=tfs.DataFrameWrapper,
        # GaussRankScaler=sk_ex.GaussRankScaler,
        # VarLenFeatureEncoder=sk_ex.VarLenFeatureEncoder,
        # MultiVarLenFeatureEncoder=sk_ex.MultiVarLenFeatureEncoder,
        # LocalizedTfidfVectorizer=sk_ex.LocalizedTfidfVectorizer,
        # TfidfEncoder=sk_ex.TfidfEncoder,
        # DatetimeEncoder=sk_ex.DatetimeEncoder,
        # FeatureGenerationTransformer=feature_generators_.FeatureGenerationTransformer,
    )
