# -*- coding:utf-8 -*-
"""

"""
import dask
import numpy as np
import pandas as pd
from dask import dataframe as dd
from scipy.stats import skew, kurtosis
from sklearn.compose import make_column_selector


class ColumnSelector(make_column_selector):
    __doc__ = make_column_selector.__doc__

    def __call__(self, df):
        if isinstance(df, dd.DataFrame):
            # # if not hasattr(df, 'iloc'):
            # #     raise ValueError("make_column_selector can only be applied to "
            # #                      "pandas dataframes")
            # df_row = df.iloc[:1]
            df_row = df

            if self.dtype_include is not None or self.dtype_exclude is not None:
                df_row = df_row.select_dtypes(include=self.dtype_include, exclude=self.dtype_exclude)
            cols = df_row.columns
            if self.pattern is not None:
                cols = cols[cols.str.contains(self.pattern, regex=True)]
            result = cols.tolist()
        else:
            result = super(ColumnSelector, self).__call__(df)

        return result

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        attrs = []
        if self.pattern:
            attrs.append(f'pattern:{self.pattern}')
        if self.dtype_include:
            attrs.append(f'include:{self.dtype_include}')
        if self.dtype_exclude:
            attrs.append(f'exclude:{self.dtype_exclude}')

        s = f'{self.__class__.__name__}({", ".join(attrs)})'
        return s


class MinMaxColumnSelector(object):
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def __call__(self, df):
        if isinstance(df, dd.DataFrame):
            return self._select_dask_dataframe(df)
        elif isinstance(df, pd.DataFrame):
            return self._select_pandas_dataframe(df)
        else:
            raise ValueError(f'Unsupported dataframe type "{type(df)}"')

    def _select_pandas_dataframe(self, df):
        if self.min is not None and self.max is not None:
            df = df.aggregate(['min', 'max'])
            df = df.loc[:, (df.loc['min'] >= self.min) & (df.loc['max'] <= self.max)]
        elif self.min is not None:
            df = df.aggregate(['min'])
            df = df.loc[:, df.loc['min'] >= self.min]
        elif self.max is not None:
            df = df.aggregate(['max'])
            df = df.loc[:, df.loc['max'] <= self.max]

        return list(df.columns)

    def _select_dask_dataframe(self, df):
        if self.min is not None and self.max is not None:
            min_values, max_values = dask.compute(df.reduction(np.min, np.min),
                                                  df.reduction(np.max, np.max))
            df = pd.DataFrame({'min': min_values, 'max': max_values
                               }).T
            df = df.loc[:, (df.loc['min'] >= self.min) & (df.loc['max'] <= self.max)]
        elif self.min is not None:
            df = pd.DataFrame({'min': df.reduction(np.min, np.min).compute()
                               }).T
            df = df.loc[:, df.loc['min'] >= self.min]
        elif self.max is not None:
            df = pd.DataFrame({'max': df.reduction(np.max, np.max).compute()
                               }).T
            df = df.loc[:, df.loc['max'] <= self.max]

        return list(df.columns)


class CompositedColumnSelector(object):
    def __init__(self, selectors):
        assert isinstance(selectors, (tuple, list)) and len(selectors) > 0
        self.selectors = selectors

    def __call__(self, df):
        n = len(self.selectors)
        for i, selector in enumerate(self.selectors):
            columns = selector(df)
            if (i == n - 1) or len(columns) == 0:
                return columns

            df = df[columns]

        return list(df.columns)  # un-reached


column_all = ColumnSelector()
column_object_category_bool = ColumnSelector(dtype_include=['object', 'category', 'bool'])
column_object = ColumnSelector(dtype_include=['object'])
column_category = ColumnSelector(dtype_include=['category'])
column_bool = ColumnSelector(dtype_include=['bool'])
column_number = ColumnSelector(dtype_include='number')
column_number_exclude_timedelta = ColumnSelector(dtype_include='number', dtype_exclude='timedelta')
column_object_category_bool_int = ColumnSelector(
    dtype_include=['object', 'category', 'bool', 'int16', 'int32', 'int64'])

column_timedelta = ColumnSelector(dtype_include='timedelta')
column_datetimetz = ColumnSelector(dtype_include='datetimetz')
column_datetime = ColumnSelector(dtype_include='datetime')
column_all_datetime = ColumnSelector(dtype_include=['datetime', 'datetimetz'])
column_int = ColumnSelector(dtype_include=['int16', 'int32', 'int64'])
column_exclude_datetime = ColumnSelector(
    dtype_exclude=['timedelta', 'datetime', 'datetimetz', 'period[M]', 'period[D]', 'period[Q]'])

column_zero_or_positive_int32 = CompositedColumnSelector(
    selectors=[column_int,
               MinMaxColumnSelector(0, np.iinfo(np.int32).max)]
)

column_positive_int32 = CompositedColumnSelector(
    selectors=[column_int,
               MinMaxColumnSelector(1, np.iinfo(np.int32).max)]
)


def column_min_max(X, min_value=None, max_value=None):
    selector = MinMaxColumnSelector(min_value, max_value)
    return selector(X)


def column_skewness_kurtosis(X, skew_threshold=0.5, kurtosis_threshold=0.5, columns=None):
    if columns is None:
        columns = column_number_exclude_timedelta(X)
    skew_values = skew(X[columns], axis=0, nan_policy='omit')
    kurtosis_values = kurtosis(X[columns], axis=0, nan_policy='omit')
    selected = [c for i, c in enumerate(columns) if
                abs(skew_values[i]) > skew_threshold or abs(kurtosis_values[i]) > kurtosis_threshold]
    return selected


def column_skewness_kurtosis_diff(X_1, X_2, diff_threshold=5, columns=None, smooth_fn=np.log, skewness_weights=1,
                                  kurtosis_weights=0):
    skew_x_1, skew_x_2, kurtosis_x_1, kurtosis_x_2, columns = calc_skewness_kurtosis(X_1, X_2, columns, smooth_fn)
    diff = np.log(
        abs(skew_x_1 - skew_x_2) * skewness_weights + np.log(abs(kurtosis_x_1 - kurtosis_x_2)) * kurtosis_weights)
    if isinstance(diff_threshold, tuple):
        index = np.argwhere((diff > diff_threshold[0]) & (diff <= diff_threshold[1]))
    else:
        index = np.argwhere(diff > diff_threshold)
    selected = [c for i, c in enumerate(columns) if i in index]
    return selected


def calc_skewness_kurtosis(X_1, X_2, columns=None, smooth_fn=np.log):
    if columns is None:
        columns = column_number_exclude_timedelta(X_1)
    X_1_t = X_1[columns]
    X_2_t = X_2[columns]
    if smooth_fn is not None:
        X_1_t[columns] = smooth_fn(X_1_t)
        X_2_t[columns] = smooth_fn(X_2_t)

    skew_x_1 = skew(X_1_t, axis=0, nan_policy='omit')
    skew_x_2 = skew(X_2_t, axis=0, nan_policy='omit')
    kurtosis_x_1 = kurtosis(X_1_t, axis=0, nan_policy='omit')
    kurtosis_x_2 = kurtosis(X_2_t, axis=0, nan_policy='omit')
    return skew_x_1, skew_x_2, kurtosis_x_1, kurtosis_x_2, columns
