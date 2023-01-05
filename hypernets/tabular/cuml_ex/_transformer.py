# -*- coding:utf-8 -*-
"""

"""
import inspect

import cudf
import cupy
import numpy as np
import pandas as pd
from cuml.common.array import CumlArray
from cuml.decomposition import TruncatedSVD
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.pipeline import Pipeline
from cuml.preprocessing import SimpleImputer, LabelEncoder, OneHotEncoder, TargetEncoder, \
    StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn import preprocessing as sk_pre, decomposition as sk_dec
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from hypernets.tabular import sklearn_ex as sk_ex
from hypernets.utils import get_params
from .. import tb_transformer


def _tf_check_n_features(self, X, reset):
    """  from cuml

    Set the `n_features_in_` attribute, or check against it.

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


class Localizable:
    def as_local(self):
        """
        convert the fitted transformer to accept pandas/numpy data, and remove cuml dependencies.
        """
        return self  # default: do nothing


def copy_attrs_as_local(tf, target, *attrs):
    from .. import CumlToolBox

    def to_local(x):
        if x is None:
            pass
        elif isinstance(x, list):
            x = list(map(to_local, x))
        elif isinstance(x, tuple):
            x = tuple(map(to_local, x))
        elif isinstance(x, dict):
            x = {ki: to_local(xi) for ki, xi in x.items()}
        elif hasattr(x, 'as_local'):
            x = x.as_local()
        else:
            x = CumlToolBox.to_local(x)[0]
        return x

    for a in attrs:
        v = getattr(tf, a)
        setattr(target, a, to_local(v))
    return target


def as_local_if_possible(tf):
    return tf.as_local() if hasattr(tf, 'as_local') else tf


def _repr(tf):
    params = get_params(tf)
    params.pop('handle', None)
    params.pop('output_type', None)
    params.pop('verbose', None)

    params = ', '.join(f'{k}={v}' for k, v in params.items())
    return f'{tf.__class__.__name__}({params})'


@tb_transformer(cudf.DataFrame, name='Pipeline')
class LocalizablePipeline(Pipeline, Localizable):
    def as_local(self):
        from sklearn.pipeline import Pipeline as SkPipeline
        steps = [(name, as_local_if_possible(tf)) for name, tf in self.steps]
        target = SkPipeline(steps, verbose=self.verbose)
        return target


@tb_transformer(cudf.DataFrame, name='StandardScaler')
class LocalizableStandardScaler(StandardScaler, Localizable):
    def as_local(self):
        target = sk_pre.StandardScaler(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)
        copy_attrs_as_local(self, target, 'scale_', 'mean_', 'var_', 'n_samples_seen_')
        return target

    # override to fix cuml
    def _check_n_features(self, X, reset):
        _tf_check_n_features(self, X, reset)


@tb_transformer(cudf.DataFrame, name='MinMaxScaler')
class LocalizableMinMaxScaler(MinMaxScaler, Localizable):
    def as_local(self):
        target = sk_pre.MinMaxScaler(self.feature_range, copy=self.copy)
        copy_attrs_as_local(self, target, 'min_', 'scale_', 'data_min_', 'data_max_', 'data_range_', 'n_samples_seen_')
        return target

    # override to fix cuml
    def _check_n_features(self, X, reset):
        _tf_check_n_features(self, X, reset)


@tb_transformer(cudf.DataFrame, name='MaxAbsScaler')
class LocalizableMaxAbsScaler(MaxAbsScaler, Localizable):
    def as_local(self):
        target = sk_pre.MaxAbsScaler(copy=self.copy)
        copy_attrs_as_local(self, target, 'scale_', 'max_abs_', 'n_samples_seen_')
        return target

    # override to fix cuml
    def _check_n_features(self, X, reset):
        _tf_check_n_features(self, X, reset)


@tb_transformer(cudf.DataFrame, name='RobustScaler')
class LocalizableRobustScaler(RobustScaler, Localizable):
    def as_local(self):
        target = sk_pre.RobustScaler(with_centering=self.with_centering, with_scaling=self.with_scaling,
                                     quantile_range=self.quantile_range, copy=self.copy)
        copy_attrs_as_local(self, target, 'scale_', 'center_', )
        return target

    # override to fix cuml
    def _check_n_features(self, X, reset):
        _tf_check_n_features(self, X, reset)


@tb_transformer(cudf.DataFrame, name='TruncatedSVD')
class LocalizableTruncatedSVD(TruncatedSVD, Localizable):
    def as_local(self):
        target = sk_dec.TruncatedSVD(self.n_components, algorithm=self.algorithm, n_iter=self.n_iter,
                                     random_state=self.random_state, tol=self.tol)
        copy_attrs_as_local(self, target, 'components_', 'explained_variance_',
                            'explained_variance_ratio_', 'singular_values_')
        return target

    def __repr__(self):
        return _repr(self)


@tb_transformer(cudf.DataFrame, name='SimpleImputer')
class LocalizableSimpleImputer(SimpleImputer, Localizable):
    def as_local(self):
        target = sk_ex.SafeSimpleImputer(missing_values=self.missing_values, strategy=self.strategy,
                                         fill_value=self.fill_value, copy=self.copy,
                                         add_indicator=self.add_indicator)
        copy_attrs_as_local(self, target, 'statistics_', 'feature_names_in_', 'n_features_in_')  # 'indicator_', )
        setattr(target, '_fit_dtype', np.dtype('float64'))

        ss = target.statistics_
        if isinstance(ss, (list, tuple)) and isinstance(ss[0], np.ndarray):
            target.statistics_ = ss[0]
        return target

    # override to fix cuml
    def _check_n_features(self, X, reset):
        _tf_check_n_features(self, X, reset)

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist() if isinstance(X, (cudf.DataFrame, pd.DataFrame)) else None

        if not isinstance(X, cudf.DataFrame):
            return super().fit(X, y)

        num_kinds = {'i', 'u', 'f'}

        if self.strategy == 'constant':
            if self.fill_value is not None:
                stat = np.full(X.shape[1], self.fill_value)
            else:
                stat = np.array([0 if d.kind in num_kinds else 'missing_value' for d in X.dtypes])
        elif self.strategy == 'most_frequent':
            mode = X.mode(dropna=True, axis=0)
            # stat = mode.iloc[0].to_pandas().values
            stat = mode.head(1).to_pandas().values.flatten()
        elif self.strategy == 'mean':
            assert all(d.kind in num_kinds for d in X.dtypes), f'only numeric type support strategy {self.strategy}'
            stat = X.mean(axis=0, skipna=True).to_pandas().values
        elif self.strategy == 'median':
            assert all(d.kind in num_kinds for d in X.dtypes), f'only numeric type support strategy {self.strategy}'
            stat = X.median(axis=0, skipna=True).to_pandas().values
        else:
            raise ValueError(f'Unsupported strategy: {self.strategy}')

        self.n_features_in_ = X.shape[1]
        self.statistics_ = stat

        return self

    def transform(self, X):
        if isinstance(X, cudf.DataFrame):
            assert self.feature_names_in_ is not None and X.columns.tolist() == self.feature_names_in_
            value = {c: v for c, v in zip(self.feature_names_in_, self.statistics_)}
            Xt = X.fillna(value)
        else:
            Xt = super().transform(X)

        if isinstance(Xt, cudf.Series):
            Xt = Xt.to_frame()
        elif isinstance(Xt, CumlArray):
            Xt = cupy.array(Xt)
        return Xt

    def __repr__(self):
        return _repr(self)


@tb_transformer(cudf.DataFrame)
class ConstantImputer(sk_ex.ConstantImputer, Localizable):
    def as_local(self):
        target = sk_ex.ConstantImputer(missing_values=self.missing_values, fill_value=self.fill_value, copy=self.copy)
        return target


@tb_transformer(cudf.DataFrame, name='OneHotEncoder')
class LocalizableOneHotEncoder(OneHotEncoder, Localizable):
    def as_local(self):
        from .. import CumlToolBox
        options = dict(categories=CumlToolBox.to_local(self.categories)[0],
                       drop=self.drop,  # sparse=self.sparse,
                       dtype=self.dtype, handle_unknown=self.handle_unknown)
        if 'sparse_output' in inspect.signature(sk_pre.OneHotEncoder.__init__).parameters.keys():
            # above sklearn 1.2
            options['sparse_output'] = self.sparse
        else:
            options['sparse'] = self.sparse
        target = sk_pre.OneHotEncoder(**options)
        copy_attrs_as_local(self, target, 'categories_', 'drop_idx_')

        try:
            if hasattr(target, '_compute_n_features_outs'):
                target._infrequent_enabled = False
                target._n_features_outs = target._compute_n_features_outs()
        except:
            pass

        return target

    def __repr__(self):
        return _repr(self)


@tb_transformer(cudf.DataFrame, name='TfidfVectorizer')
class LocalizableTfidfVectorizer(TfidfVectorizer, Localizable):
    def as_local(self):
        from .. import CumlToolBox
        target = sk_ex.TfidfVectorizer(
            # input="content",
            # encoding="utf-8",
            # decode_error="ignore",
            # strip_accents=None,
            lowercase=self.lowercase,
            preprocessor=self.preprocessor,
            tokenizer=None,
            analyzer=self.analyzer,
            stop_words=self.stop_words,
            # token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=self.ngram_range,
            max_df=self.max_df,
            min_df=self.min_df,
            max_features=self.max_features,
            vocabulary=self.vocabulary,
            binary=self.binary,
            dtype=np.float64,
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
        )

        # copy_attrs_as_local(self, target, 'idf_', 'vocabulary_', '_fixed_vocabulary', 'stop_words_')
        copy_attrs_as_local(self, target, 'vocabulary_', 'stop_words_')
        target.fixed_vocabulary_ = self._fixed_vocabulary
        if isinstance(target.vocabulary_, pd.Series):
            target.vocabulary_ = {v: i for i, v in target.vocabulary_.to_dict().items()}  # to dict,and swap key/value
        idf = self.idf_
        if len(idf.shape) > 1 and idf.shape[0] == 1:
            idf = idf[0]
        target.idf_ = CumlToolBox.to_local(idf)[0]
        return target

    def __repr__(self):
        return _repr(self)


@tb_transformer(cudf.DataFrame)
class TfidfEncoder(sk_ex.TfidfEncoder, Localizable):
    def create_encoder(self):
        return LocalizableTfidfVectorizer(**self.encoder_kwargs)

    def as_local(self):
        target = sk_ex.TfidfEncoder(columns=self.columns, flatten=self.flatten, **self.encoder_kwargs)
        copy_attrs_as_local(self, target, 'encoders_')
        return target


@tb_transformer(cudf.DataFrame)
class DatetimeEncoder(sk_ex.DatetimeEncoder, Localizable):
    all_items = sk_ex.DatetimeEncoder.all_items.copy()
    all_items.pop('week')  # does not support

    default_include = [k for k in sk_ex.DatetimeEncoder.default_include if k != 'week']

    @staticmethod
    def to_dataframe(X):
        if isinstance(X, cudf.DataFrame):
            pass
        elif isinstance(X, cupy.ndarray):
            X = cudf.DataFrame(X)
        elif isinstance(X, CumlArray):
            X = X.to_output('cudf')
        else:
            X = sk_ex.DatetimeEncoder.to_dataframe(X)
        return X

    def as_local(self):
        target = sk_ex.DatetimeEncoder(columns=self.columns, include=self.include,
                                       exclude=self.exclude, extra=self.extra,
                                       drop_constants=self.drop_constants)
        copy_attrs_as_local(self, target, 'extract_')
        return target


_te_stub = TargetEncoder()


@tb_transformer(cudf.DataFrame)
class SlimTargetEncoder(TargetEncoder, BaseEstimator):
    """
    The slimmed TargetEncoder with 'train' and 'train_encode' attribute were set to None.
    """

    def __init__(self, n_folds=4, smooth=0, seed=42, split_method='interleaved', dtype=None, output_2d=False):
        super().__init__(n_folds=n_folds, smooth=smooth, seed=seed, split_method=split_method)

        self.dtype = dtype
        self.output_2d = output_2d

    def fit(self, X, y, **kwargs):
        super().fit(X, y)
        self.train = None
        self.train_encode = None
        return self

    def fit_transform(self, X, y, **kwargs):
        sig = inspect.signature(self._fit_transform)
        if 'fold_ids' in sig.parameters.keys():
            Xt, _ = self._fit_transform(X, y, None)
        else:
            Xt, _ = self._fit_transform(X, y)

        self.train = None
        self.train_encode = None
        self._fitted = True
        if self.dtype is not None:
            Xt = Xt.astype(self.dtype)
        if self.output_2d:
            Xt = Xt.reshape(-1, 1)
        return Xt

    def transform(self, X):
        Xt = super().transform(X)
        if self.dtype is not None:
            Xt = Xt.astype(self.dtype)
        if self.output_2d:
            Xt = Xt.reshape(-1, 1)
        return Xt

    def _check_is_fitted(self):
        check_is_fitted(self, '_fitted')

    def _is_train_df(self, df):
        return False

    def as_local(self):
        target = sk_ex.SlimTargetEncoder(n_folds=self.n_folds, smooth=self.smooth, seed=self.seed,
                                         split_method=self.split_method, dtype=self.dtype, output_2d=self.output_2d)
        copy_attrs_as_local(self, target, '_fitted', 'train', 'train_encode', 'encode_all', 'mean', 'output_2d')
        return target

    if not hasattr(_te_stub, 'split_method'):
        @property
        def split_method(self):
            return self.split


@tb_transformer(cudf.DataFrame)
class MultiTargetEncoder(sk_ex.MultiTargetEncoder):
    target_encoder_cls = SlimTargetEncoder
    label_encoder_cls = LabelEncoder

    def as_local(self):
        target = sk_ex.MultiTargetEncoder(n_folds=self.n_folds, smooth=self.smooth, seed=self.seed,
                                          split_method=self.split_method)
        target.encoders_ = {k: le.as_local() for k, le in self.encoders_.items()}
        return target


@tb_transformer(cudf.DataFrame, name='LabelEncoder')
class LocalizableLabelEncoder(LabelEncoder, Localizable):
    def as_local(self):
        target = sk_pre.LabelEncoder()
        copy_attrs_as_local(self, target, 'classes_', )
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
        copy_attrs_as_local(self, target, 'classes_', )
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
