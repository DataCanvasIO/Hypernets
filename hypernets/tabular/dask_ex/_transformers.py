import re
from collections import defaultdict

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml import preprocessing as dm_pre, decomposition as dm_dec
from sklearn import preprocessing as sk_pre
from sklearn.base import BaseEstimator, TransformerMixin

from hypernets.utils import logging

logger = logging.get_logger(__name__)


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        assert len(X.shape) == 2

        if isinstance(X, (pd.DataFrame, dd.DataFrame)):
            return self._fit_df(X, y)
        elif isinstance(X, (np.ndarray, da.Array)):
            return self._fit_array(X, y)
        else:
            raise Exception(f'Unsupported type "{type(X)}"')

    def _fit_df(self, X, y=None):
        return self._fit_array(X.values, y.values if y else None)

    def _fit_array(self, X, y=None):
        n_features = X.shape[1]
        for n in range(n_features):
            le = dm_pre.LabelEncoder()
            le.fit(X[:, n])
            self.encoders[n] = le
        return self

    def transform(self, X):
        assert len(X.shape) == 2

        if isinstance(X, (dd.DataFrame, pd.DataFrame)):
            return self._transform_dask_df(X)
        elif isinstance(X, (da.Array, np.ndarray)):
            return self._transform_dask_array(X)
        else:
            raise Exception(f'Unsupported type "{type(X)}"')

    def _transform_dask_df(self, X):
        data = self._transform_dask_array(X.values)

        if isinstance(X, dd.DataFrame):
            result = dd.from_dask_array(data, columns=X.columns)
        else:
            result = pd.DataFrame(data, columns=X.columns)
        return result

    def _transform_dask_array(self, X):
        n_features = X.shape[1]
        assert n_features == len(self.encoders.items())

        data = []
        for n in range(n_features):
            data.append(self.encoders[n].transform(X[:, n]))

        if isinstance(X, da.Array):
            result = da.stack(data, axis=-1, allow_unknown_chunksizes=True)
        else:
            result = np.stack(data, axis=-1)

        return result

    # def fit_transform(self, X, y=None):
    #     return self.fit(X, y).transform(X)


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


class TruncatedSVD(dm_dec.TruncatedSVD):
    def fit_transform(self, X, y=None):
        X_orignal = X
        if isinstance(X, pd.DataFrame):
            X = dd.from_pandas(X, npartitions=2)

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


class MaxAbsScaler(sk_pre.MaxAbsScaler):
    __doc__ = sk_pre.MaxAbsScaler.__doc__

    def fit(self, X, y=None, ):
        from dask_ml.utils import handle_zeros_in_scale

        self._reset()
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().fit(X, y)

        max_abs = X.reduction(lambda x: x.abs().max(),
                              aggregate=lambda x: x.max(),
                              token=self.__class__.__name__
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


class SafeOrdinalEncoder(BaseEstimator, TransformerMixin):
    __doc__ = r'Adapted from dask_ml OrdinalEncoder\n' + dm_pre.OrdinalEncoder.__doc__

    def __init__(self, columns=None, dtype=np.float64):
        self.columns = columns
        self.dtype = dtype

    def fit(self, X, y=None):
        """Determine the categorical columns to be encoded.

        Parameters
        ----------
        X : pandas.DataFrame or dask.dataframe.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        self.columns_ = X.columns
        self.dtypes_ = {c: X[c].dtype for c in X.columns}

        if self.columns is None:
            columns = X.select_dtypes(include=["category", 'object', 'bool']).columns
        else:
            columns = self.columns

        X = X.categorize(columns=columns)

        self.categorical_columns_ = columns
        self.non_categorical_columns_ = X.columns.drop(self.categorical_columns_)
        self.categories_ = {c: X[c].cat.categories.sort_values() for c in columns}

        return self

    def transform(self, X, y=None):
        """Ordinal encode the categorical columns in X

        Parameters
        ----------
        X : pd.DataFrame or dd.DataFrame
        y : ignored

        Returns
        -------
        transformed : pd.DataFrame or dd.DataFrame
            Same type as the input
        """
        if not X.columns.equals(self.columns_):
            raise ValueError(
                "Columns of 'X' do not match the training "
                "columns. Got {!r}, expected {!r}".format(X.columns, self.columns)
            )

        encoder = self.make_encoder(self.categorical_columns_, self.categories_, self.dtype)
        if isinstance(X, pd.DataFrame):
            X = encoder(X)
        elif isinstance(X, dd.DataFrame):
            X = X.map_partitions(encoder)
        else:
            raise TypeError("Unexpected type {}".format(type(X)))

        return X

    def inverse_transform(self, X, missing_value=None):
        """Inverse ordinal-encode the columns in `X`

        Parameters
        ----------
        X : array or dataframe
            Either the NumPy, dask, or pandas version

        missing_value : skip doc

        Returns
        -------
        data : DataFrame
            Dask array or dataframe will return a Dask DataFrame.
            Numpy array or pandas dataframe will return a pandas DataFrame
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)
        elif isinstance(X, da.Array):
            # later on we concat(..., axis=1), which requires
            # known divisions. Suboptimal, but I think unavoidable.
            unknown = np.isnan(X.chunks[0]).any()
            if unknown:
                lengths = da.blockwise(len, "i", X[:, 0], "i", dtype="i8").compute()
                X = X.copy()
                chunks = (tuple(lengths), X.chunks[1])
                X._chunks = chunks
            X = dd.from_dask_array(X, columns=self.columns_)

        decoder = self.make_decoder(self.categorical_columns_, self.categories_, self.dtypes_)

        if isinstance(X, dd.DataFrame):
            X = X.map_partitions(decoder)
        else:
            X = decoder(X)

        return X

    @staticmethod
    def make_encoder(columns, categories, dtype):
        mappings = {}
        for col in columns:
            cat = categories[col]
            unseen = len(cat)
            m = defaultdict(dtype)
            for k, v in zip(cat, range(unseen)):
                m[k] = dtype(v + 1)
            mappings[col] = m

        def encode_column(x, c):
            return mappings[c][x]

        def safe_ordinal_encoder(pdf):
            assert isinstance(pdf, pd.DataFrame)

            pdf = pdf.copy()
            vf = np.vectorize(encode_column, excluded='c', otypes=[dtype])
            for col in columns:
                r = vf(pdf[col].values, col)
                if r.dtype != dtype:
                    # print(r.dtype, 'astype', dtype)
                    r = r.astype(dtype)
                pdf[col] = r
            return pdf

        return safe_ordinal_encoder

    @staticmethod
    def make_decoder(columns, categories, dtypes):
        def decode_column(x, col):
            cat = categories[col]
            xi = int(x)
            unseen = cat.shape[0]  # len(cat)
            if unseen >= xi >= 1:
                return cat[xi - 1]
            else:
                dtype = dtypes[col]
                if dtype in (np.float32, np.float64, np.float):
                    return np.nan
                elif dtype in (np.int32, np.int64, np.int, np.uint32, np.uint64, np.uint):
                    return -1
                else:
                    return None

        def safe_ordinal_decoder(pdf):
            assert isinstance(pdf, pd.DataFrame)

            pdf = pdf.copy()
            for col in columns:
                vf = np.vectorize(decode_column, excluded='col', otypes=[dtypes[col]])
                pdf[col] = vf(pdf[col].values, col)
            return pdf

        return safe_ordinal_decoder


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
