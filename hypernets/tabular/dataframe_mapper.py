# -*- coding:utf-8 -*-
"""
Adapted from: https://github.com/scikit-learn-contrib/sklearn-pandas
1. Fix the problem of confusion of column names
2. Support `columns` is a callable object
"""
import contextlib
import sys

import numpy as np
import pandas as pd
import six
from dask import array as da
from dask import dataframe as dd
from scipy import sparse as _sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import _name_estimators, Pipeline
from sklearn.utils import tosequence

from hypernets.utils import logging
from . import dask_ex as dex

logger = logging.get_logger(__name__)


def _call_fit(fit_method, X, y=None, **kwargs):
    """
    helper function, calls the fit or fit_transform method with the correct
    number of parameters

    fit_method: fit or fit_transform method of the transformer
    X: the data to fit
    y: the target vector relative to X, optional
    kwargs: any keyword arguments to the fit method

    return: the result of the fit or fit_transform method

    WARNING: if this function raises a TypeError exception, test the fit
    or fit_transform method passed to it in isolation as _call_fit will not
    distinguish TypeError due to incorrect number of arguments from
    other TypeError
    """
    try:
        return fit_method(X, y, **kwargs)
    except TypeError:
        # fit takes only one argument
        return fit_method(X, **kwargs)


class TransformerPipeline(Pipeline):
    """
    Pipeline that expects all steps to be transformers taking a single X
    argument, an optional y argument, and having fit and transform methods.

    Code is copied from sklearn's Pipeline
    """

    def __init__(self, steps):
        names, estimators = zip(*steps)
        if len(dict(steps)) != len(steps):
            raise ValueError(
                "Provided step names are not unique: %s" % (names,))

        # shallow copy of steps
        self.steps = tosequence(steps)
        estimator = estimators[-1]

        for e in estimators:
            if (not (hasattr(e, "fit") or hasattr(e, "fit_transform")) or not
            hasattr(e, "transform")):
                raise TypeError("All steps of the chain should "
                                "be transforms and implement fit and transform"
                                " '%s' (type %s) doesn't)" % (e, type(e)))

        if not hasattr(estimator, "fit"):
            raise TypeError("Last step of chain should implement fit "
                            "'%s' (type %s) doesn't)"
                            % (estimator, type(estimator)))

    def _pre_transform(self, X, y=None, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Xt = _call_fit(transform.fit_transform,
                               Xt, y, **fit_params_steps[name])
            else:
                Xt = _call_fit(transform.fit,
                               Xt, y, **fit_params_steps[name]).transform(Xt)
        return Xt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        Xt, fit_params = self._pre_transform(X, y, **fit_params)
        _call_fit(self.steps[-1][-1].fit, Xt, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        Xt, fit_params = self._pre_transform(X, y, **fit_params)
        if hasattr(self.steps[-1][-1], 'fit_transform'):
            return _call_fit(self.steps[-1][-1].fit_transform,
                             Xt, y, **fit_params)
        else:
            return _call_fit(self.steps[-1][-1].fit,
                             Xt, y, **fit_params).transform(Xt)


def make_transformer_pipeline(*steps):
    """Construct a TransformerPipeline from the given estimators.
    """
    return TransformerPipeline(_name_estimators(steps))


PY3 = sys.version_info[0] == 3
if PY3:
    string_types = text_type = str
else:
    string_types = basestring  # noqa
    text_type = unicode  # noqa


def _handle_feature(fea):
    """
    Convert 1-dimensional arrays to 2-dimensional column vectors.
    """
    if len(fea.shape) == 1:
        if isinstance(fea, dd.DataFrame):
            fea = da.stack([fea], axis=-1)
        else:
            fea = np.array([fea]).T

    return fea


def _build_transformer(transformers):
    if isinstance(transformers, list):
        transformers = make_transformer_pipeline(*transformers)
    return transformers


def _build_feature(columns, transformers, options={}):
    return (columns, _build_transformer(transformers), options)


def _get_feature_names(estimator, columns=None):
    """
    Attempt to extract feature names based on a given estimator
    """
    if hasattr(estimator, 'classes_'):
        return estimator.classes_
    elif hasattr(estimator, 'get_feature_names'):
        return estimator.get_feature_names(columns)
    return None


@contextlib.contextmanager
def add_column_names_to_exception(column_names):
    # Stolen from https://stackoverflow.com/a/17677938/356729
    try:
        yield
    except Exception as ex:
        if ex.args:
            msg = u'{}: {}'.format(column_names, ex.args[0])
        else:
            msg = text_type(column_names)
        ex.args = (msg,) + ex.args[1:]
        raise


class DataFrameMapper(BaseEstimator, TransformerMixin):
    """
    Map Pandas data frame column subsets to their own
    sklearn transformation.
    """

    def __init__(self, features, default=False, sparse=False, df_out=False,
                 input_df=False, df_out_dtype_transforms=None):
        """
        Params:

        features    a list of tuples with features definitions.
                    The first element is the pandas column selector. This can
                    be a string (for one column) or a list of strings.
                    The second element is an object that supports
                    sklearn's transform interface, or a list of such objects.
                    The third element is optional and, if present, must be
                    a dictionary with the options to apply to the
                    transformation. Example: {'alias': 'day_of_week'}

        default     default transformer to apply to the columns not
                    explicitly selected in the mapper. If False (default),
                    discard them. If None, pass them through untouched. Any
                    other transformer will be applied to all the unselected
                    columns as a whole, taken as a 2d-array.

        sparse      will return sparse matrix if set True and any of the
                    extracted features is sparse. Defaults to False.

        df_out      return a pandas data frame, with each column named using
                    the pandas column that created it (if there's only one
                    input and output) or the input columns joined with '_'
                    if there's multiple inputs, and the name concatenated with
                    '_1', '_2' etc if there's multiple outputs. NB: does not
                    work if *default* or *sparse* are true

        input_df    If ``True`` pass the selected columns to the transformers
                    as a pandas DataFrame or Series. Otherwise pass them as a
                    numpy array. Defaults to ``False``.
        """
        if df_out and (sparse or default):
            raise ValueError("Can not use df_out with sparse or default")

        self.features = features
        self.default = default
        self.sparse = sparse
        self.df_out = df_out
        self.input_df = input_df
        self.df_out_dtype_transforms = df_out_dtype_transforms

        # fitted
        self.fitted_features_ = None

    @staticmethod
    def _build(features, default):
        if isinstance(features, list):
            built_features = [_build_feature(*f) for f in features]
        else:
            built_features = features
        built_default = _build_transformer(default)

        return built_features, built_default

    def _get_col_subset(self, X, cols, input_df=False):
        """
        Get a subset of columns from the given table X.

        X       a Pandas dataframe; the table to select columns from
        cols    a string or list of strings representing the columns
                to select

        Returns a numpy array with the data from the selected columns
        """
        if isinstance(cols, string_types):
            return_vector = True
            cols = [cols]
        else:
            return_vector = False

        # Needed when using the cross-validation compatibility
        # layer for sklearn<0.16.0.
        # Will be dropped on sklearn-pandas 2.0.
        if isinstance(X, list):
            X = [x[cols] for x in X]
            X = pd.DataFrame(X)

        # elif isinstance(X, DataWrapper):
        #     X = X.df  # fetch underlying data

        if return_vector:
            t = X[cols[0]]
        else:
            t = X[cols]

        # return either a DataFrame/Series or a numpy array
        if input_df:
            return t
        else:
            return t.values

    def fit(self, X, y=None):
        built_features, built_default = self._build(self.features, self.default)

        fitted_features = []
        selected_columns = []

        for columns_def, transformers, options in built_features:
            logger.debug(f'columns:({columns_def}), transformers:({transformers}), options:({options})')
            if callable(columns_def):
                columns = columns_def(X)
            elif isinstance(columns_def, string_types):
                columns = [columns_def]
            else:
                columns = columns_def

            if isinstance(columns, (list, tuple)):
                columns = [c for c in columns if c not in selected_columns]

            fitted_features.append((columns, transformers, options))
            if columns is None or len(columns) <= 0:
                continue

            selected_columns += columns
            if transformers is not None:
                input_df = options.get('input_df', self.input_df)
                with add_column_names_to_exception(columns):
                    Xt = self._get_col_subset(X, columns, input_df)
                    _call_fit(transformers.fit, Xt, y)
                    # print(f'{transformers}:{Xt.dtypes}')

        # handle features not explicitly selected
        if built_default is not False and len(X.columns) > len(selected_columns):
            unselected_columns = [c for c in X.columns.to_list() if c not in selected_columns]
            if built_default is not None:
                with add_column_names_to_exception(unselected_columns):
                    Xt = self._get_col_subset(X, unselected_columns, self.input_df)
                    _call_fit(built_default.fit, Xt, y)
            fitted_features.append((unselected_columns, built_default, {}))

        self.fitted_features_ = fitted_features

        return self

    def transform(self, X):
        selected_columns = []
        transformed_columns = []
        extracted = []

        for columns, transformers, options in self.fitted_features_:
            if columns is None or len(columns) < 1:
                continue
            selected_columns += columns

            input_df = options.get('input_df', self.input_df)
            alias = options.get('alias')

            Xt = self._get_col_subset(X, columns, input_df)
            if transformers is not None:
                with add_column_names_to_exception(columns):
                    # print(f'before ---- {transformers}:{Xt.dtypes}')
                    Xt = transformers.transform(Xt)
                    # print(f'after ---- {transformers}:{pd.DataFrame(Xt).dtypes}')

            extracted.append(_handle_feature(Xt))

            transformed_columns += self.get_names(columns, transformers, Xt, alias)

        return self._to_transform_result(X, extracted, transformed_columns)

    def fit_transform(self, X, y=None, *fit_args):
        fitted_features = []
        selected_columns = []
        transformed_columns = []
        extracted = []

        built_features, built_default = self._build(self.features, self.default)
        for columns_def, transformers, options in built_features:
            if callable(columns_def):
                columns = columns_def(X)
            elif isinstance(columns_def, string_types):
                columns = [columns_def]
            else:
                columns = columns_def
            if isinstance(columns, (list, tuple)) and len(set(selected_columns).intersection(set(columns))) > 0:
                columns = [c for c in columns if c not in selected_columns]

            if columns is None or len(columns) < 1:
                continue

            fitted_features.append((columns, transformers, options))
            selected_columns += columns
            if logger.is_debug_enabled():
                logger.debug(f'fit_transform {len(columns)} columns with:\n{transformers}')

            input_df = options.get('input_df', self.input_df)
            alias = options.get('alias')

            Xt = self._get_col_subset(X, columns, input_df)
            if transformers is not None:
                with add_column_names_to_exception(columns):
                    if hasattr(transformers, 'fit_transform'):
                        Xt = _call_fit(transformers.fit_transform, Xt, y)
                    else:
                        _call_fit(transformers.fit, Xt, y)
                        Xt = transformers.transform(Xt)

            extracted.append(_handle_feature(Xt))
            if logger.is_debug_enabled():
                logger.debug(f'columns:{len(columns)}')
            transformed_columns += self.get_names(columns, transformers, Xt, alias)
            if logger.is_debug_enabled():
                logger.debug(f'transformed_names_:{len(transformed_columns)}')

        # handle features not explicitly selected
        if built_default is not False and len(X.columns) > len(selected_columns):
            unselected_columns = [c for c in X.columns.to_list() if c not in selected_columns]
            Xt = self._get_col_subset(X, unselected_columns, self.input_df)
            if built_default is not None:
                with add_column_names_to_exception(unselected_columns):
                    if hasattr(built_default, 'fit_transform'):
                        Xt = _call_fit(built_default.fit_transform, Xt, y)
                    else:
                        _call_fit(built_default.fit, Xt, y)
                        Xt = built_default.transform(Xt)
                transformed_columns += self.get_names(unselected_columns, built_default, Xt)
            else:
                # if not applying a default transformer, keep column names unmodified
                transformed_columns += unselected_columns
            extracted.append(_handle_feature(Xt))

            fitted_features.append((unselected_columns, built_default, {}))

        self.fitted_features_ = fitted_features

        return self._to_transform_result(X, extracted, transformed_columns)

    def get_names(self, columns, transformer, x, alias=None):
        """
        Return verbose names for the transformed columns.

        columns       name (or list of names) of the original column(s)
        transformer   transformer - can be a TransformerPipeline
        x             transformed columns (numpy.ndarray)
        alias         base name to use for the selected columns
        """
        # logger.debug(
        #     f'get_names: {isinstance(columns, list)}, len(columns):{len(columns)} columns:{columns}, alias:{alias}')
        if alias is not None:
            name = alias
        elif isinstance(columns, list):
            name = '_'.join(map(str, columns))
        else:
            name = columns
        num_cols = x.shape[1] if len(x.shape) > 1 else 1
        if num_cols > 1:
            # If there are as many columns as classes in the transformer,
            # infer column names from classes names.

            # If we are dealing with multiple transformers for these columns
            # attempt to extract the names from each of them, starting from the
            # last one
            # logger.debug(f'transformer:{transformer}')
            if isinstance(transformer, (TransformerPipeline, Pipeline)):
                inverse_steps = transformer.steps[::-1]
                estimators = (estimator for name, estimator in inverse_steps)
                names_steps = (_get_feature_names(e, columns) for e in estimators)
                names = next((n for n in names_steps if n is not None), None)

                if names is None and len(columns) == num_cols:
                    names = list(columns)
            # Otherwise use the only estimator present
            else:
                names = _get_feature_names(transformer, columns)

            if logger.is_debug_enabled():
                # logger.debug(f'names:{names}')
                logger.debug(f'names:{len(names)}')
            if names is not None and len(names) == num_cols:
                return list(names)  # ['%s_%s' % (name, o) for o in names]
            # otherwise, return name concatenated with '_1', '_2', etc.
            else:
                return [name + '_' + str(o) for o in range(num_cols)]
        else:
            return [name]

    def get_dtypes(self, extracted):
        dtypes_features = [self.get_dtype(ex) for ex in extracted]
        return [dtype for dtype_feature in dtypes_features
                for dtype in dtype_feature]

    def get_dtype(self, ex):
        if isinstance(ex, (np.ndarray, da.Array)) or _sparse.issparse(ex):
            return [ex.dtype] * ex.shape[1]
        elif isinstance(ex, (pd.DataFrame, dd.DataFrame)):
            return list(ex.dtypes)
        else:
            raise TypeError(type(ex))

    def _to_transform_result(self, X, extracted, transformed_columns):
        # combine the feature outputs into one array. at this point we lose track of which features
        # were created from which input columns, so it's assumed that that doesn't matter to the model.

        # no data transformed, raise a error
        if extracted is None or len(extracted) == 0:
            raise ValueError("No data output, ??? ")

        # If any of the extracted features is sparse, combine sparsely.
        # Otherwise, combine as normal arrays.
        if isinstance(X, dd.DataFrame):
            extracted = [a.values if isinstance(a, dd.DataFrame) else a for a in extracted]
            stacked = dex.hstack_array(extracted)
        elif any(_sparse.issparse(fea) for fea in extracted):
            stacked = _sparse.hstack(extracted).tocsr()
            # return a sparse matrix only if the mapper was initialized
            # with sparse=True
            if not self.sparse:
                stacked = stacked.toarray()
        else:
            stacked = np.hstack(extracted)

        if self.df_out:
            # if no rows were dropped preserve the original index,
            # otherwise use a new integer one
            if isinstance(X, dd.DataFrame):
                df_out = dd.from_dask_array(stacked, columns=transformed_columns, index=None)
            else:
                no_rows_dropped = len(X) == len(stacked)
                index = X.index if no_rows_dropped else None
                df_out = pd.DataFrame(stacked, columns=transformed_columns, index=index)

            # output different data types, if appropriate
            dtypes = self.get_dtypes(extracted)

            # preserve types
            for col, dtype, stype in zip(transformed_columns, dtypes, df_out.dtypes.tolist()):
                if dtype != stype:
                    if logger.is_debug_enabled():
                        logger.debug(f'convert {col} as {dtype} from {stype}')
                    df_out[col] = df_out[col].astype(dtype)
            df_out = self._dtype_transform(df_out)
            return df_out
        else:
            return stacked

    def _dtype_transform(self, df_out):
        if self.df_out_dtype_transforms is not None:
            for columns, dtype in self.df_out_dtype_transforms:
                if callable(columns):
                    columns = columns(df_out)
                if isinstance(columns, list) and len(columns) <= 0:
                    continue
                df_out[columns] = df_out[columns].astype(dtype)
        return df_out
