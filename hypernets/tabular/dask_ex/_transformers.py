import re
from collections import defaultdict
from functools import partial

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml import preprocessing as dm_pre, decomposition as dm_dec
from sklearn import preprocessing as sk_pre
from sklearn.base import BaseEstimator, TransformerMixin

from hypernets.tabular import sklearn_ex as skex
from hypernets.tabular import tb_transformer
from hypernets.utils import logging, const

logger = logging.get_logger(__name__)


#
# class MultiLabelEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.encoders = {}
#
#     def fit(self, X, y=None):
#         assert len(X.shape) == 2
#
#         if isinstance(X, (pd.DataFrame, dd.DataFrame)):
#             return self._fit_df(X, y)
#         elif isinstance(X, (np.ndarray, da.Array)):
#             return self._fit_array(X, y)
#         else:
#             raise Exception(f'Unsupported type "{type(X)}"')
#
#     def _fit_df(self, X, y=None):
#         return self._fit_array(X.values, y.values if y else None)
#
#     def _fit_array(self, X, y=None):
#         n_features = X.shape[1]
#         for n in range(n_features):
#             le = dm_pre.LabelEncoder()
#             le.fit(X[:, n])
#             self.encoders[n] = le
#         return self
#
#     def transform(self, X):
#         assert len(X.shape) == 2
#
#         if isinstance(X, (dd.DataFrame, pd.DataFrame)):
#             return self._transform_dask_df(X)
#         elif isinstance(X, (da.Array, np.ndarray)):
#             return self._transform_dask_array(X)
#         else:
#             raise Exception(f'Unsupported type "{type(X)}"')
#
#     def _transform_dask_df(self, X):
#         data = self._transform_dask_array(X.values)
#
#         if isinstance(X, dd.DataFrame):
#             result = dd.from_dask_array(data, columns=X.columns)
#         else:
#             result = pd.DataFrame(data, columns=X.columns)
#         return result
#
#     def _transform_dask_array(self, X):
#         n_features = X.shape[1]
#         assert n_features == len(self.encoders.items())
#
#         data = []
#         for n in range(n_features):
#             data.append(self.encoders[n].transform(X[:, n]))
#
#         if isinstance(X, da.Array):
#             result = da.stack(data, axis=-1, allow_unknown_chunksizes=True)
#         else:
#             result = np.stack(data, axis=-1)
#
#         return result
#
#     # def fit_transform(self, X, y=None):
#     #     return self.fit(X, y).transform(X)

@tb_transformer(dd.DataFrame)
class SafeOneHotEncoder(dm_pre.OneHotEncoder):
    def fit(self, X, y=None):
        if isinstance(X, (dd.DataFrame, pd.DataFrame)) and self.categories == "auto" \
                and any(d.name in {'object', 'bool'} for d in X.dtypes):
            a = []
            if isinstance(X, dd.DataFrame):
                for i in range(len(X.columns)):
                    Xi = X.iloc[:, i]
                    if Xi.dtype.name in {'object', 'bool'}:
                        Xi = Xi.astype('category').cat.as_known()
                    a.append(Xi)
                X = dd.concat(a, axis=1, ignore_unknown_divisions=True)
            else:
                for i in range(len(X.columns)):
                    Xi = X.iloc[:, i]
                    if Xi.dtype.name in {'object', 'bool'}:
                        Xi = Xi.astype('category')
                    a.append(Xi)
                X = pd.concat(a, axis=1)

        return super(SafeOneHotEncoder, self).fit(X, y)

    def get_feature_names(self, input_features=None):
        """
        Override this method to remove non-alphanumeric chars
        """
        # if not hasattr(self, 'drop_idx_'):
        #     setattr(self, 'drop_idx_', None)
        # return super(SafeOneHotEncoder, self).get_feature_names(input_features)

        # check_is_fitted(self)
        cats = self.categories_
        if input_features is None:
            input_features = ['x%d' % i for i in range(len(cats))]
        elif len(input_features) != len(self.categories_):
            raise ValueError(
                "input_features should have length equal to number of "
                "features ({}), got {}".format(len(self.categories_),
                                               len(input_features)))

        feature_names = []
        for i in range(len(cats)):
            names = [input_features[i] + '_' + str(idx) + '_' + re.sub('[^A-Za-z0-9_]+', '_', str(t))
                     for idx, t in enumerate(cats[i])]
            # if self.drop_idx_ is not None and self.drop_idx_[i] is not None:
            #     names.pop(self.drop_idx_[i])
            feature_names.extend(names)

        return np.array(feature_names, dtype=object)


@tb_transformer(dd.DataFrame)
class TruncatedSVD(dm_dec.TruncatedSVD):
    def fit_transform(self, X, y=None):
        X_orignal = X
        if isinstance(X, pd.DataFrame):
            X = dd.from_pandas(X, npartitions=2).clear_divisions()

        if isinstance(X, dd.DataFrame):
            # y = y.values.compute_chunk_sizes() if y is not None else None
            r = super(TruncatedSVD, self).fit_transform(X.values.compute_chunk_sizes(), None)
        else:
            r = super(TruncatedSVD, self).fit_transform(X, y)

        if isinstance(X_orignal, (pd.DataFrame, np.ndarray)):
            r = r.compute()
        return r

    def transform(self, X, y=None):
        if isinstance(X, dd.DataFrame):
            return super(TruncatedSVD, self).transform(X.values, y)

        return super(TruncatedSVD, self).transform(X, y)

    def inverse_transform(self, X):
        if isinstance(X, dd.DataFrame):
            return super(TruncatedSVD, self).inverse_transform(X.values)

        return super(TruncatedSVD, self).inverse_transform(X)


@tb_transformer(dd.DataFrame)
class MaxAbsScaler(sk_pre.MaxAbsScaler):
    __doc__ = sk_pre.MaxAbsScaler.__doc__

    def fit(self, X, y=None, ):
        from dask_ml.utils import handle_zeros_in_scale

        self._reset()
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().fit(X, y)

        max_abs = X.reduction(lambda x: x.abs().max(),
                              aggregate=lambda x: x.max(),
                              token=f'{self.__class__.__name__}.fit'
                              ).compute()
        scale = handle_zeros_in_scale(max_abs)

        setattr(self, 'max_abs_', max_abs)
        setattr(self, 'scale_', scale)
        setattr(self, 'n_samples_seen_', 0)

        self.n_features_in_ = X.shape[1]
        return self

    def partial_fit(self, X, y=None, ):
        raise NotImplementedError()

    def transform(self, X, y=None, copy=None, ):
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().transform(X)

        # Workaround for https://github.com/dask/dask/issues/2840
        if isinstance(X, dd.DataFrame):
            X = X.div(self.scale_)
        else:
            X = X / self.scale_
        return X

    def inverse_transform(self, X, y=None, copy=None, ):
        if not hasattr(self, "scale_"):
            raise Exception(
                "This %(name)s instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before "
                "using this method."
            )

        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().inverse_transform(X)

        if copy:
            X = X.copy()
        if isinstance(X, dd.DataFrame):
            X = X.mul(self.scale_)
        else:
            X = X * self.scale_

        return X


def _safe_ordinal_encoder(categories, dtype, pdf):
    assert isinstance(pdf, pd.DataFrame)

    def encode_column(x, c):
        return c[x]

    mappings = {}
    for col, cat in categories.items():
        unseen = len(cat)
        m = defaultdict(dtype)
        for k, v in zip(cat, range(unseen)):
            m[k] = dtype(v + 1)
        mappings[col] = m

    pdf = pdf.copy()
    vf = np.vectorize(encode_column, excluded='c', otypes=[dtype])
    for col, m in mappings.items():
        r = vf(pdf[col].values, m)
        if r.dtype != dtype:
            # print(r.dtype, 'astype', dtype)
            r = r.astype(dtype)
        pdf[col] = r
    return pdf


def _safe_ordinal_decoder(categories, dtypes, pdf):
    assert isinstance(pdf, pd.DataFrame)

    def decode_column(x, col):
        cat = categories[col]
        xi = int(x)
        unseen = cat.shape[0]  # len(cat)
        if unseen >= xi >= 1:
            return cat[xi - 1]
        else:
            dtype = dtypes[col]
            if dtype in (np.float32, np.float64, float):
                return np.nan
            elif dtype in (np.int32, np.int64, np.uint32, np.uint64, np.uint, int):
                return -1
            else:
                return None

    pdf = pdf.copy()
    for col in categories.keys():
        vf = np.vectorize(decode_column, excluded='col', otypes=[dtypes[col]])
        pdf[col] = vf(pdf[col].values, col)
    return pdf


@tb_transformer(dd.DataFrame)
class SafeOrdinalEncoder(BaseEstimator, TransformerMixin):
    __doc__ = r'Adapted from dask_ml OrdinalEncoder\n' + dm_pre.OrdinalEncoder.__doc__

    def __init__(self, columns=None, dtype=np.float64):
        self.columns = columns
        self.dtype = dtype

        # fitted
        self.columns_ = None
        self.dtypes_ = None
        self.categorical_columns_ = None
        self.non_categorical_columns_ = None
        self.categories_ = None

    def fit(self, X, y=None):
        self.columns_ = X.columns.to_list()
        self.dtypes_ = {c: X[c].dtype for c in X.columns}

        if self.columns is None:
            columns = X.select_dtypes(include=["category", 'object', 'bool']).columns.to_list()
        else:
            columns = self.columns

        X = X.categorize(columns=columns)

        self.categorical_columns_ = columns
        self.non_categorical_columns_ = X.columns.drop(self.categorical_columns_)
        self.categories_ = {c: X[c].cat.categories.sort_values() for c in columns}

        return self

    def transform(self, X, y=None):
        if X.columns.to_list() != self.columns_:
            raise ValueError(
                "Columns of 'X' do not match the training "
                "columns. Got {!r}, expected {!r}".format(X.columns.to_list(), self.columns)
            )

        encoder = self.make_encoder(self.categories_, self.dtype)
        if isinstance(X, pd.DataFrame):
            X = encoder(X)
        elif isinstance(X, dd.DataFrame):
            X = X.map_partitions(encoder)
        else:
            raise TypeError("Unexpected type {}".format(type(X)))

        return X

    def inverse_transform(self, X, missing_value=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.categorical_columns_)
        elif isinstance(X, da.Array):
            # later on we concat(..., axis=1), which requires
            # known divisions. Suboptimal, but I think unavoidable.
            unknown = np.isnan(X.chunks[0]).any()
            if unknown:
                lengths = da.blockwise(len, "i", X[:, 0], "i", dtype="i8").compute()
                X = X.copy()
                chunks = (tuple(lengths), X.chunks[1])
                X._chunks = chunks
            X = dd.from_dask_array(X, columns=self.categorical_columns_)

        decoder = self.make_decoder(self.categories_, self.dtypes_)

        if isinstance(X, dd.DataFrame):
            X = X.map_partitions(decoder)
        else:
            X = decoder(X)

        return X

    @staticmethod
    def make_encoder(categories, dtype):
        return partial(_safe_ordinal_encoder, categories, dtype)

    @staticmethod
    def make_decoder(categories, dtypes):
        return partial(_safe_ordinal_decoder, categories, dtypes)


class DataInterceptEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, fit=False, fit_transform=False, transform=False, inverse_transform=False):
        self._intercept_fit = fit
        self._intercept_fit_transform = fit_transform
        self._intercept_transform = transform
        self._intercept_inverse_transform = inverse_transform

        super(DataInterceptEncoder, self).__init__()

    def fit(self, X, *args, **kwargs):
        if self._intercept_fit:
            self.intercept(X, *args, **kwargs)

        return self

    def fit_transform(self, X, *args, **kwargs):
        if self._intercept_fit_transform:
            X = self.intercept(X, *args, **kwargs)

        return X

    def transform(self, X, *args, **kwargs):
        if self._intercept_transform:
            X = self.intercept(X, *args, **kwargs)

        return X

    def inverse_transform(self, X, *args, **kwargs):
        if self._intercept_inverse_transform:
            X = self.intercept(X, *args, **kwargs)

        return X

    def intercept(self, X, *args, **kwargs):
        raise NotImplementedError()


class CallableAdapterEncoder(DataInterceptEncoder):
    def __init__(self, fn, **kwargs):
        assert callable(fn)

        self.fn = fn

        super(CallableAdapterEncoder, self).__init__(**kwargs)

    def intercept(self, X, *args, **kwargs):
        return self.fn(X, *args, **kwargs)


@tb_transformer(dd.DataFrame)
class DataCacher(DataInterceptEncoder):
    """
    persist and cache dask dataframe and array
    """

    def __init__(self, cache_dict, cache_key, remove_keys=None, **kwargs):
        assert isinstance(cache_dict, dict)

        if isinstance(remove_keys, str):
            remove_keys = set(remove_keys.split(','))

        self._cache_dict = cache_dict
        self.cache_key = cache_key
        self.remove_keys = remove_keys

        super(DataCacher, self).__init__(**kwargs)

    def intercept(self, X, *args, **kwargs):
        if self.cache_key:
            if isinstance(X, (dd.DataFrame, da.Array)):
                if logger.is_debug_enabled():
                    logger.debug(f'persist and cache {X._name} as {self.cache_key}')

                X = X.persist()

            self._cache_dict[self.cache_key] = X

        if self.remove_keys:
            for key in self.remove_keys:
                if key in self._cache_dict.keys():
                    if logger.is_debug_enabled():
                        logger.debug(f'remove cache {key}')
                    del self._cache_dict[key]

        return X

    @property
    def cache_dict(self):
        return list(self._cache_dict.keys())


@tb_transformer(dd.DataFrame)
class CacheCleaner(DataInterceptEncoder):

    def __init__(self, cache_dict, **kwargs):
        assert isinstance(cache_dict, dict)

        self._cache_dict = cache_dict

        super(CacheCleaner, self).__init__(**kwargs)

    def intercept(self, X, *args, **kwargs):
        if logger.is_debug_enabled():
            logger.debug(f'clean cache with {list(self._cache_dict.keys())}')
        self._cache_dict.clear()

        return X

    @property
    def cache_dict(self):
        return list(self._cache_dict.keys())

    # # override this to remove 'cache_dict' from estimator __expr__
    # @classmethod
    # def _get_param_names(cls):
    #     params = super()._get_param_names()
    #     return [p for p in params if p != 'cache_dict']


@tb_transformer(dd.DataFrame)
class DataFrameWrapper(skex.DataFrameWrapper):

    def transform(self, X):
        transformed = self.transformer.transform(X)

        if isinstance(transformed, da.Array):
            transformed = dd.from_dask_array(transformed, columns=self.columns)
        elif isinstance(transformed, np.ndarray):
            transformed = pd.DataFrame(transformed, columns=self.columns)
        else:
            transformed.columns = self.columns

        return transformed


class AdaptedTransformer(BaseEstimator):
    """
    Adapt sklearn style transformer to support dask data objects.
    Note: adapted transformer uses locale resource only.
    """

    def __init__(self, cls, attributes, *args, **kwargs):
        super(AdaptedTransformer, self).__init__()

        assert isinstance(cls, type)
        assert attributes is None or isinstance(attributes, (tuple, list, set))

        self.adapted_transformer_ = cls(*args, **kwargs)
        self.adapted_attributes_ = set(attributes) if attributes is not None else None

    def fit(self, X, y=None, **kwargs):
        if y is None:
            self._adapt(self.adapted_transformer_.fit, X, **kwargs)
        else:
            self._adapt(self.adapted_transformer_.fit, X, y, **kwargs)
        return self

    def transform(self, X, *args, **kwargs):
        return self._adapt(self.adapted_transformer_.transform, X, *args, **kwargs)

    def inverse_transform(self, X, *args, **kwargs):
        return self._adapt(self.adapted_transformer_.inverse_transform, X, *args, **kwargs)

    def fit_transform(self, X, y=None, **kwargs):
        transformer = self.adapted_transformer_

        if hasattr(transformer, 'fit_transform'):
            if y is None:
                return self._adapt(transformer.fit_transform, X, **kwargs)
            else:
                return self._adapt(transformer.fit_transform, X, y, **kwargs)
        else:
            self.fit(X, y, **kwargs)
            return self._adapt(transformer.transform, X, **kwargs)

    @staticmethod
    def _adapt(fn, X, *args, **kwargs):
        is_dask_frame_X = isinstance(X, (dd.DataFrame, dd.Series))
        is_dask_array_X = isinstance(X, da.Array)

        if is_dask_frame_X:
            npartitions = X.npartitions
            X = X.compute()
        elif is_dask_array_X:
            chunks = X.chunks
            X = X.compute()

        args = [a.compute() if isinstance(a, (dd.DataFrame, dd.Series, da.Array)) else a for a in args]
        r = fn(X, *args, **kwargs)

        if isinstance(r, (pd.DataFrame, pd.Series, np.ndarray)):
            if is_dask_frame_X:
                r = dd.from_pandas(r, npartitions=npartitions).clear_divisions()
            elif is_dask_array_X:
                r = da.from_array(r, chunks=chunks)

        return r

    def __getattribute__(self, name: str):
        if name in {'adapted_attributes_', 'adapted_transformer_'}:
            return super().__getattribute__(name)

        adapted_attributes = self.adapted_attributes_
        if adapted_attributes is not None and name in adapted_attributes:
            adapted_transformer = self.adapted_transformer_
            return getattr(adapted_transformer, name)

        return super().__getattribute__(name)

    def __dir__(self):
        adapted_attributes = self.adapted_attributes_
        if adapted_attributes is not None:
            return set(list(adapted_attributes) + list(super().__dir__()))

        return super().__dir__()


@tb_transformer(dd.DataFrame)
class LgbmLeavesEncoder(AdaptedTransformer, TransformerMixin):

    def __init__(self, cat_vars, cont_vars, task, **params):
        attributes = {'lgbm', 'cat_vars', 'cont_vars', 'new_columns', 'task', 'lgbm_params'}
        super(LgbmLeavesEncoder, self).__init__(skex.LgbmLeavesEncoder, attributes,
                                                cat_vars, cont_vars, task, **params)


@tb_transformer(dd.DataFrame)
class CategorizeEncoder(skex.CategorizeEncoder):

    def fit(self, X, y=None):
        super(CategorizeEncoder, self).fit(X, y)

        if len(self.new_columns) > 0:
            nuniques = [nunique for _, _, nunique in self.new_columns]
            nuniques = dask.compute(*nuniques)

            new_columns = [(col, dtype, nunique)
                           for (col, dtype, _), nunique in zip(self.new_columns, nuniques)]
            self.new_columns = new_columns

        return self


@tb_transformer(dd.DataFrame)
class MultiKBinsDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, bins=None, strategy='quantile', dtype='float64'):
        assert strategy in {'uniform', 'quantile'}
        assert dtype is None or dtype.startswith('int') or dtype.startswith('float')

        super(MultiKBinsDiscretizer, self).__init__()

        self.columns = columns
        self.bins = bins
        self.strategy = strategy
        self.dtype = dtype

        # fitted
        self.new_columns = []
        self.bin_edges_ = {}

    def fit(self, X, y=None):
        assert isinstance(X, (dd.DataFrame, pd.DataFrame))
        is_dask_X = isinstance(X, dd.DataFrame)

        new_columns = []
        if self.columns is None:
            self.columns = X.select_dtypes(['float', 'float64', 'int', 'int64']).columns.tolist()

            # define new_columns
        for col in self.columns:
            new_name = col + const.COLUMNNAME_POSTFIX_DISCRETE
            c_bins = self.bins
            if c_bins is None or c_bins <= 0:
                n_unique = X.loc[:, col].nunique()
                if is_dask_X:
                    n_unique = n_unique.compute()
                c_bins = int(round(n_unique ** 0.25)) + 1
            new_columns.append((col, new_name, c_bins))

        # compute bin_borders
        if self.strategy == 'quantile':
            bin_edges = {}
            for col, _, c_bins in new_columns:
                step = 1.0 / c_bins
                qs = [i * step for i in range(1, int(c_bins))]
                q = X[col].quantile(qs)
                if is_dask_X:
                    q = q.compute()
                bin_edges[col] = q.to_list()
        else:  # strategy == 'uniform'
            X = X[self.columns]
            mns, mxs = X.min(), X.max()
            if is_dask_X:
                mns, mxs = dask.compute(mns, mxs)
            bin_edges = {col: [mns[col], mxs[col]] for col in self.columns}

        self.new_columns = new_columns
        self.bin_edges_ = bin_edges

        return self

    def transform(self, X):
        assert isinstance(X, (dd.DataFrame, pd.DataFrame))

        if self.strategy == 'quantile':
            transformer = MultiKBinsDiscretizer._transform_df_quantile
        else:  # strategy == 'uniform'
            transformer = MultiKBinsDiscretizer._transform_df_uniform

        if isinstance(X, dd.DataFrame):
            meta = X.dtypes.to_dict()
            dtype = self.dtype if self.dtype is not None else 'int'
            for _, new_name, _ in self.new_columns:
                meta[new_name] = dtype
            fn = partial(transformer, self.new_columns, self.bin_edges_, self.dtype)
            X = X.map_partitions(fn, meta=meta)
        else:
            X = transformer(self.new_columns, self.bin_edges_, self.dtype, X)

        return X

    @staticmethod
    def _transform_df_quantile(new_columns, bin_edges, dtype, X):
        for col, new_name, c_bins in new_columns:
            edges = bin_edges[col]
            X[new_name] = np.searchsorted(edges, X[col], side='left')

            if dtype is not None and not str(dtype).startswith('int'):
                X[new_name] = X[new_name].astype(dtype)

        return X

    @staticmethod
    def _transform_df_uniform(new_columns, bin_edges, dtype, X):
        for col, new_name, c_bins in new_columns:
            mn, mx = bin_edges[col]
            if dtype is not None and str(dtype).startswith('float'):
                if mx > mn and c_bins > 1:
                    step = (mx - mn) / c_bins
                    fn = lambda v: 0. if v <= mn else c_bins - 1. if v >= mx else float(int((v - mn) / step))
                else:
                    fn = lambda v: float(v > mn)
            else:
                if mx > mn and c_bins > 1:
                    step = (mx - mn) / c_bins
                    fn = lambda v: 0 if v <= mn else c_bins - 1 if v >= mx else int((v - mn) / step)
                else:
                    fn = lambda v: int(v > mn)

            X[new_name] = X[col].map(fn)

        return X


@tb_transformer(dd.DataFrame)
class MultiVarLenFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Example:
        features = [ ('feature_1','|'), ... ]
        encoder = MultiVarLenFeatureEncoder(features)
        ...
    """

    def __init__(self, features):
        assert isinstance(features, (tuple, list))
        assert all([isinstance(fs, (tuple, list)) and len(fs) > 1 for fs in features])
        assert all([f[0] for f in features]), 'Feature name is required.'
        assert all([f[1] for f in features]), 'Separator is required.'

        super(MultiVarLenFeatureEncoder, self).__init__()

        self.seps_ = {feature[0]: feature[1] for feature in features}

        # fitted
        self.keys_ = None
        self.max_length_ = {}  # key: feature, value: max length

    def fit(self, X, y=None):
        assert isinstance(X, (dd.DataFrame, pd.DataFrame))

        is_dask_X = isinstance(X, dd.DataFrame)
        feature_keys = {}
        max_length = {}

        sep_to_features = defaultdict(list)
        for f, sep in self.seps_.items():
            sep_to_features[sep].append(f)

        for sep, features in sep_to_features.items():
            if is_dask_X:
                meta = {f: 'str' for f in features}
                fn_chunk = partial(MultiVarLenFeatureEncoder._fit_part, sep)
                fn_agg = partial(MultiVarLenFeatureEncoder._agg_part, sep)
                t = X[features].reduction(fn_chunk, aggregate=fn_agg, meta=meta,
                                          token=f'{self.__class__.__name__}.fit'
                                          ).compute()
            else:  # pd.DataFrame
                t = self._fit_part(sep, X[features])

            assert isinstance(t, pd.Series)
            for f, aggregated in t.to_dict().items():
                a = aggregated.split(sep)
                max_length[f] = int(a[0])
                feature_keys[f] = a[1:]

        self.max_length_ = max_length
        self.keys_ = feature_keys

        return self

    def transform(self, X):
        assert isinstance(X, (dd.DataFrame, pd.DataFrame))
        assert len(set(self.seps_.keys()) - set(X.columns.to_list())) == 0, \
            f'Not found {set(self.seps_.keys()) - set(X.columns.to_list())} in X'

        if isinstance(X, dd.DataFrame):
            meta = X.dtypes.to_dict()
            for f in self.seps_.keys():
                meta[f] = 'object'
            fn = partial(MultiVarLenFeatureEncoder._transform_df, self.seps_, self.keys_, self.max_length_)
            X = X.map_partitions(fn, meta=meta)
        else:
            X = self._transform_df(self.seps_, self.keys_, self.max_length_, X)

        return X

    @staticmethod
    def _fit_part(sep, X):
        assert isinstance(X, pd.DataFrame)
        fn = partial(MultiVarLenFeatureEncoder._aggregate_key, sep, False)
        return X.agg(fn, axis=0)

    @staticmethod
    def _agg_part(sep, X):
        assert isinstance(X, pd.DataFrame)
        fn = partial(MultiVarLenFeatureEncoder._aggregate_key, sep, True)
        return X.agg(fn, axis=0)

    @staticmethod
    def _aggregate_key(sep, y_include_max_len, y):
        key_set = set()
        max_len = 0

        for v in y:
            a = list(filter(len, v.split(sep)))
            if y_include_max_len:
                len_a = int(a[0])
                a = a[1:]
            else:
                len_a = len(a)
            if len_a > max_len:
                max_len = len_a
            key_set.update(a)
        key_set = list(key_set)
        key_set.sort()

        return sep.join([str(max_len)] + key_set)

    @staticmethod
    def _transform_df(feature_seps, feature_keys, feature_max_length, X):
        for f, sep in feature_seps.items():
            keys = feature_keys.get(f, [])
            max_len = feature_max_length.get(f, 0)
            assert isinstance(keys, (list, tuple)) and isinstance(max_len, int)
            if len(keys) <= 1 or max_len < 1:
                continue
            X[f] = MultiVarLenFeatureEncoder._encode(sep, keys, max_len, X[f])

        return X

    @staticmethod
    def _encode(sep, keys, max_len, y):
        unseen = len(keys) + 1
        key_values = dict(zip(keys, list(range(1, unseen + 1))))
        result = np.empty(len(y), dtype=object)

        for yi, v in enumerate(y):
            result_yi = []
            arr = filter(len, v.split(sep))
            for vi, k in enumerate(arr):
                result_yi.append(key_values[k] if k in key_values else unseen)
                if vi >= max_len - 1:
                    break
            while len(result_yi) < max_len:
                result_yi.append(0)
            result[yi] = result_yi

        return result


@tb_transformer(dd.DataFrame)
class LocalizedTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=None, max_df=1.0, min_df=1):
        super(LocalizedTfidfVectorizer, self).__init__()

        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df

        # fitted
        self.vocabulary_ = None

    def fit(self, X, y=None, **kwargs):
        chunk_kwargs = dict(max_features=self.max_features, max_df=self.max_df, min_df=self.min_df)

        if isinstance(X, dd.Series):
            voc_df = X.reduction(self.fit_part, aggregate=self.agg_part, chunk_kwargs=chunk_kwargs, meta=(None, 'f8'))
            voc_df = voc_df.compute()
        elif isinstance(X, (pd.Series, np.ndarray)):
            voc_df = self.fit_part(X, **chunk_kwargs)
        else:
            raise ValueError(f'Unsupported data type: {type(X).__name__}')

        vocabulary = {v: i for v, i in zip(voc_df.index, range(len(voc_df)))}
        self.vocabulary_ = vocabulary

        return self

    def transform(self, X, y=None):
        if isinstance(X, dd.Series):
            meta = {f'x{i}': 'f8' for _, i in self.vocabulary_.items()}
            r = X.map_partitions(self.transform_part, self.vocabulary_, meta=meta)
        elif isinstance(X, (pd.Series, np.ndarray)):
            r = self.transform_part(X, self.vocabulary_)
        else:
            raise ValueError(f'Unsupported data type: {type(X).__name__}')

        return r

    @staticmethod
    def fit_part(part, max_features=None, max_df=None, min_df=None):
        t = skex.LocalizedTfidfVectorizer(use_idf=False, max_features=max_features, max_df=max_df, min_df=min_df)
        t.fit(part)
        return pd.Series(t.vocabulary_)

    @staticmethod
    def agg_part(part):
        return pd.Series(index=part.columns, dtype='f8')

    @staticmethod
    def transform_part(part, voc):
        t = skex.LocalizedTfidfVectorizer(use_idf=False, vocabulary=voc)
        t.fit(np.array(['']))
        result = t.transform(part).toarray()

        if isinstance(part, pd.Series):
            columns = [f'x{i}' for _, i in voc.items()]
            result = pd.DataFrame(result, index=part.index, columns=columns)

        return result
