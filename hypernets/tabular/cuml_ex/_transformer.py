# -*- coding:utf-8 -*-
"""

"""
import cudf
import cupy
import numpy as np
import pandas as pd
from cuml.common.array import CumlArray
from cuml.decomposition import TruncatedSVD
from cuml.pipeline import Pipeline
from cuml.preprocessing import SimpleImputer, LabelEncoder, OneHotEncoder, TargetEncoder, \
    StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn import preprocessing as sk_pre, impute as sk_imp, decomposition as sk_dec
from sklearn.base import BaseEstimator, TransformerMixin

from hypernets.tabular import sklearn_ex as sk_ex
from .. import tb_transformer


class Localizable:
    def as_local(self):
        """
        convert the fitted transformer to accept pandas/numpy data, and remove cuml dependencies.
        """
        return self  # default: do nothing


def copy_attrs(tf, target, *attrs):
    from .. import CumlToolBox
    for p in attrs:
        v = getattr(tf, p)
        v, = CumlToolBox.to_local(v)
        setattr(target, p, v)
    return target


@tb_transformer(cudf.DataFrame, name='Pipeline')
class LocalizablePipeline(Pipeline, Localizable):
    def as_local(self):
        from sklearn.pipeline import Pipeline as SkPipeline
        steps = [(name, tf.as_local()) for name, tf in self.steps]
        target = SkPipeline(steps, verbose=self.verbose)
        return target


@tb_transformer(cudf.DataFrame, name='StandardScaler')
class LocalizableStandardScaler(StandardScaler, Localizable):
    def as_local(self):
        target = sk_pre.StandardScaler(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)
        copy_attrs(self, target, 'scale_', 'mean_', 'var_', 'n_samples_seen_')
        return target


@tb_transformer(cudf.DataFrame, name='MinMaxScaler')
class LocalizableMinMaxScaler(MinMaxScaler, Localizable):
    def as_local(self):
        target = sk_pre.MinMaxScaler(self.feature_range, copy=self.copy)
        copy_attrs(self, target, 'min_', 'scale_', 'data_min_', 'data_max_', 'data_range_', 'n_samples_seen_')
        return target


@tb_transformer(cudf.DataFrame, name='MaxAbsScaler')
class LocalizableMaxAbsScaler(MaxAbsScaler, Localizable):
    def as_local(self):
        target = sk_pre.MaxAbsScaler(copy=self.copy)
        copy_attrs(self, target, 'scale_', 'max_abs_', 'n_samples_seen_')
        return target


@tb_transformer(cudf.DataFrame, name='RobustScaler')
class LocalizableRobustScaler(RobustScaler, Localizable):
    def as_local(self):
        target = sk_pre.RobustScaler(with_centering=self.with_centering, with_scaling=self.with_scaling,
                                     quantile_range=self.quantile_range, copy=self.copy)
        copy_attrs(self, target, 'scale_', 'center_', )
        return target


@tb_transformer(cudf.DataFrame, name='TruncatedSVD')
class LocalizableTruncatedSVD(TruncatedSVD, Localizable):
    def as_local(self):
        target = sk_dec.TruncatedSVD(self.n_components, algorithm=self.algorithm, n_iter=self.n_iter,
                                     random_state=self.random_state, tol=self.tol)
        copy_attrs(self, target, 'components_', 'explained_variance_', 'explained_variance_ratio_', 'singular_values_')
        return target


@tb_transformer(cudf.DataFrame, name='SimpleImputer')
class LocalizableSimpleImputer(SimpleImputer, Localizable):
    def as_local(self):
        target = sk_imp.SimpleImputer(missing_values=self.missing_values, strategy=self.strategy,
                                      fill_value=self.fill_value, copy=self.copy, add_indicator=self.add_indicator)
        copy_attrs(self, target, 'statistics_', )  # 'indicator_', )

        ss = target.statistics_
        if isinstance(ss, (list, tuple)) and isinstance(ss[0], np.ndarray):
            target.statistics_ = ss[0]
        return target

    # override to fix cuml
    def _check_n_features(self, X, reset):
        """Set the `n_features_in_` attribute, or check against it.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to `X.shape[1]`.
            Else, the attribute must already exist and the function checks
            that it is equal to `X.shape[1]`.
        """
        n_features = X.shape[1]

        if reset:
            self.n_features_in_ = n_features
        else:
            if not hasattr(self, 'n_features_in_'):
                raise RuntimeError(
                    "The reset parameter is False but there is no "
                    "n_features_in_ attribute. Is this estimator fitted?"
                )
            if n_features != self.n_features_in_:
                raise ValueError(
                    'X has {} features, but this {} is expecting {} features '
                    'as input.'.format(n_features, self.__class__.__name__,
                                       self.n_features_in_)
                )

    def transform(self, X):
        Xt = super().transform(X)
        if isinstance(Xt, cudf.Series):
            Xt = Xt.to_frame()
        elif isinstance(Xt, CumlArray):
            Xt = cupy.array(Xt)
        return Xt


@tb_transformer(cudf.DataFrame)
class ConstantImputer(BaseEstimator, TransformerMixin, Localizable):
    def __init__(self, missing_values=np.nan, fill_value=None, copy=True) -> None:
        super().__init__()

        self.missing_values = missing_values
        self.fill_value = fill_value
        self.copy = copy

    def fit(self, X, y=None, ):
        return self

    def transform(self, X, y=None):
        if self.copy:
            X = X.copy()

        X.replace(self.missing_values, self.fill_value, inplace=True)
        return X


@tb_transformer(cudf.DataFrame, name='OneHotEncoder')
class LocalizableOneHotEncoder(OneHotEncoder, Localizable):
    def as_local(self):
        from .. import CumlToolBox
        target = sk_pre.OneHotEncoder(categories=CumlToolBox.to_local(self.categories)[0],
                                      drop=self.drop, sparse=self.sparse,
                                      dtype=self.dtype, handle_unknown=self.handle_unknown)
        copy_attrs(self, target, 'categories_', 'drop_idx_')
        return target


@tb_transformer(cudf.DataFrame, name='TargetEncoder')
class LocalizableTargetEncoder(TargetEncoder, Localizable):
    def as_local(self):
        target = sk_ex.TargetEncoder(n_folds=self.n_folds, smooth=self.smooth, seed=self.seed, split_method=self.split)
        copy_attrs(self, target, '_fitted', 'train', 'train_encode', 'encode_all', 'mean')
        return target


@tb_transformer(cudf.DataFrame, name='LabelEncoder')
class LocalizableLabelEncoder(LabelEncoder, Localizable):
    def as_local(self):
        target = sk_pre.LabelEncoder()
        copy_attrs(self, target, 'classes_', )
        return target

    # override to accept pd.Series and ndarray
    def inverse_transform(self, y) -> cudf.Series:
        if isinstance(y, pd.Series):
            y = cudf.from_pandas(y)
        elif isinstance(y, np.ndarray):
            y = cudf.from_pandas(pd.Series(y))
        elif isinstance(y, cupy.ndarray):
            y = cudf.Series(y)

        return super().inverse_transform(y)


@tb_transformer(cudf.DataFrame)
class SafeLabelEncoder(LabelEncoder):
    def __init__(self, *, verbose=False, output_type=None):
        super().__init__(handle_unknown='ignore', verbose=verbose, output_type=output_type)

    def fit_transform(self, y: cudf.Series, z=None) -> cudf.Series:
        t = super().fit_transform(y, z=z)
        return t

    def transform(self, y: cudf.Series) -> cudf.Series:
        t = super().transform(y)
        t.fillna(len(self.classes_), inplace=True)
        return t

    def as_local(self):
        target = sk_ex.SafeLabelEncoder()
        copy_attrs(self, target, 'classes_', )
        return target


@tb_transformer(cudf.DataFrame)
class MultiLabelEncoder(BaseEstimator, Localizable):
    def __init__(self, columns=None, dtype=None):
        super().__init__()

        self.columns = columns
        self.dtype = dtype

        # fitted
        self.encoders = {}

    def fit(self, X: cudf.DataFrame, y=None):
        assert isinstance(X, cudf.DataFrame)

        if self.columns is None:
            self.columns = X.columns.tolist()

        for col in self.columns:
            data = X.loc[:, col]
            if data.dtype == 'object':
                data = data.astype('str')
            le = SafeLabelEncoder()
            le.fit(data)
            self.encoders[col] = le

        return self

    def transform(self, X: cudf.DataFrame):
        assert isinstance(X, cudf.DataFrame) and self.columns is not None
        others = [c for c in X.columns.tolist() if c not in self.columns]

        dfs = []
        if len(others) > 0:
            dfs.append(X[others])

        for col in self.columns:
            data = X.loc[:, col]
            if data.dtype == 'object':
                data = data.astype('str')
            t = self.encoders[col].transform(data)
            if self.dtype is not None:
                t = t.astype(self.dtype)
            dfs.append(t)

        df = cudf.concat(dfs, axis=1, ignore_index=True) if len(dfs) > 1 else dfs[0]
        df.index = X.index
        df.columns = others + self.columns
        if len(others) > 0:
            df = df[X.columns]

        return df

    def fit_transform(self, X: cudf.DataFrame, *args):
        if self.columns is None:
            self.columns = X.columns.tolist()
            others = []
        else:
            others = [c for c in X.columns.tolist() if c not in self.columns]

        dfs = []
        if len(others) > 0:
            dfs.append(X[others])

        for col in self.columns:
            data = X.loc[:, col]
            if data.dtype == 'object':
                data = data.astype('str')
            le = SafeLabelEncoder()
            t = le.fit_transform(data)  # .to_frame(name=col)
            if self.dtype is not None:
                t = t.astype(self.dtype)
            dfs.append(t)
            self.encoders[col] = le
        df = cudf.concat(dfs, axis=1, ignore_index=True) if len(dfs) > 1 else dfs[0]
        df.index = X.index
        df.columns = others + self.columns
        if len(others) > 0:
            df = df[X.columns]

        return df

    def as_local(self):
        target = sk_ex.MultiLabelEncoder()
        target.columns = self.columns
        target.dtype = self.dtype
        target.encoders = {k: e.as_local() for k, e in self.encoders.items()}
        return target
