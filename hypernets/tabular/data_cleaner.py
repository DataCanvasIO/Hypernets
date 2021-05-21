# -*- coding:utf-8 -*-
"""

"""
import copy

import dask
import numpy as np
import pandas as pd
from dask import dataframe as dd

from hypernets.utils import logging
from .column_selector import column_object, column_int, column_object_category_bool_int

logger = logging.get_logger(__name__)


def _reduce_mem_usage(df, verbose=True):
    """
    Adaption from :https://blog.csdn.net/xckkcxxck/article/details/88170281
    :param verbose:
    :return:
    """
    if isinstance(df, dd.DataFrame):
        raise Exception('"reduce_mem_usage" is not supported for Dask DataFrame.')

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))


def _drop_duplicated_columns(X):
    if isinstance(X, dd.DataFrame):
        duplicates = X.reduction(chunk=lambda c: pd.DataFrame(c.T.duplicated()).T,
                                 aggregate=lambda a: np.all(a, axis=0)).compute()
    else:
        duplicates = X.T.duplicated()

    dup_cols = [i for i, v in duplicates.items() if v]
    columns = [c for c in X.columns.to_list() if c not in dup_cols]
    X = X[columns]
    return X, dup_cols


def _correct_object_dtype(X):
    object_columns = column_object(X)

    if isinstance(X, dd.DataFrame):
        def detect_dtype(df):
            result = {}
            df = df.copy()
            for col in object_columns:
                try:
                    df[col] = df[col].astype('float')
                    result[col] = [True]  # float-able
                except:
                    result[col] = [False]
            return pd.DataFrame(result)

        floatable = X.reduction(chunk=detect_dtype,
                                aggregate=lambda a: np.all(a, axis=0)).compute()
        float_columns = [i for i, v in floatable.items() if v]
        for col in float_columns:
            X[col] = X[col].astype('float')
        logger.debug(f'Correct columns [{",".join(float_columns)}] to float.')
    else:
        for col in object_columns:
            try:
                X[col] = X[col].astype('float')
            except Exception as e:
                logger.debug(f'Correct object column [{col}] failed. {e}')

    return X


def _drop_constant_columns(X):
    if isinstance(X, dd.DataFrame):
        nunique = X.reduction(chunk=lambda c: pd.DataFrame(c.nunique(dropna=True)).T,
                              aggregate=np.max).compute()
    else:
        nunique = X.nunique(dropna=True)

    const_cols = [i for i, v in nunique.items() if v <= 1]
    columns = [c for c in X.columns.to_list() if c not in const_cols]
    X = X[columns]
    return X, const_cols


def _drop_idness_columns(X):
    cols = column_object_category_bool_int(X)
    if len(cols) <= 0:
        return X, []
    X_ = X[cols]
    if isinstance(X_, dd.DataFrame):
        nunique = X_.reduction(chunk=lambda c: pd.DataFrame(c.nunique(dropna=True)).T,
                               aggregate=np.max)
        rows = X_.reduction(lambda df: df.shape[0], np.sum)
        nunique, rows = dask.compute(nunique, rows)
    else:
        nunique = X_.nunique(dropna=True)
        rows = X_.shape[0]

    droped = [i for i, v in nunique.items() if v / rows > 0.99]
    columns = [c for c in X.columns.to_list() if c not in droped]
    X = X[columns]
    return X, droped


class DataCleaner:
    def __init__(self, nan_chars=None, correct_object_dtype=True, drop_constant_columns=True,
                 drop_duplicated_columns=False, drop_label_nan_rows=True, drop_idness_columns=True,
                 replace_inf_values=np.nan, drop_columns=None, reduce_mem_usage=False, int_convert_to='float'):
        self.nan_chars = nan_chars
        self.correct_object_dtype = correct_object_dtype
        self.drop_constant_columns = drop_constant_columns
        self.drop_label_nan_rows = drop_label_nan_rows
        self.drop_idness_columns = drop_idness_columns
        self.replace_inf_values = replace_inf_values
        self.drop_columns = drop_columns
        self.drop_duplicated_columns = drop_duplicated_columns
        self.reduce_mem_usage = reduce_mem_usage
        self.int_convert_to = int_convert_to
        self.df_meta_ = None
        self.columns_ = None

        self.dropped_constant_columns_ = None
        self.dropped_idness_columns_ = None
        self.dropped_duplicated_columns_ = None

    def get_params(self):
        return {
            'nan_chars': self.nan_chars,
            'correct_object_dtype': self.correct_object_dtype,
            'drop_constant_columns': self.drop_constant_columns,
            'drop_label_nan_rows': self.drop_label_nan_rows,
            'drop_idness_columns': self.drop_idness_columns,
            # 'replace_inf_values': self.replace_inf_values,
            'drop_columns': self.drop_columns,
            'drop_duplicated_columns': self.drop_duplicated_columns,
            'reduce_mem_usage': self.reduce_mem_usage,
            'int_convert_to': self.int_convert_to
        }

    def _drop_columns(self, X, cols):
        if cols is None or len(cols) <= 0:
            return X
        X = X[[c for c in X.columns.to_list() if c not in cols]]
        return X

    def clean_data(self, X, y):
        assert isinstance(X, (pd.DataFrame, dd.DataFrame))
        y_name = '__tabular-toolbox__Y__'

        if y is not None:
            X[y_name] = y

        if self.nan_chars is not None:
            logger.debug(f'replace chars{self.nan_chars} to NaN')
            X = X.replace(self.nan_chars, np.nan)

        if y is not None:
            if self.drop_label_nan_rows:
                logger.debug('clean the rows which label is NaN')
                X = X.dropna(subset=[y_name])
            y = X.pop(y_name)

        if self.correct_object_dtype:
            logger.debug('correct data type for object columns.')
            # for col in column_object(X):
            #     try:
            #         X[col] = X[col].astype('float')
            #     except Exception as e:
            #         logger.error(f'Correct object column [{col}] failed. {e}')
            X = _correct_object_dtype(X)

        if self.drop_duplicated_columns:
            logger.debug('drop duplicated columns')
            if self.dropped_duplicated_columns_ is not None:
                X = self._drop_columns(X, self.dropped_duplicated_columns_)
            else:
                X, self.dropped_duplicated_columns_ = _drop_duplicated_columns(X)

        if self.drop_idness_columns:
            logger.debug('drop idness columns')
            if self.dropped_idness_columns_ is not None:
                X = self._drop_columns(X, self.dropped_idness_columns_)
            else:
                X, self.dropped_idness_columns_ = _drop_idness_columns(X)

        if self.int_convert_to is not None:
            logger.debug(f'convert int type to {self.int_convert_to}')
            int_cols = column_int(X)
            X[int_cols] = X[int_cols].astype(self.int_convert_to)

        if self.drop_columns is not None:
            logger.debug(f'drop columns:{self.drop_columns}')
            for col in self.drop_columns:
                X.pop(col)

        if self.drop_constant_columns:
            logger.debug('drop invalidate columns')
            if self.dropped_constant_columns_ is not None:
                X = self._drop_columns(X, self.dropped_constant_columns_)
            else:
                X, self.dropped_constant_columns_ = _drop_constant_columns(X)

        o_cols = column_object(X)
        X[o_cols] = X[o_cols].astype('str')

        return X, y

    def fit_transform(self, X, y=None, copy_data=True):
        if copy_data:
            X = copy.deepcopy(X)
            if y is not None:
                y = copy.deepcopy(y)

        X, y = self.clean_data(X, y)
        if self.reduce_mem_usage:
            logger.debug('reduce memory usage')
            _reduce_mem_usage(X)

        if self.replace_inf_values is not None:
            logger.debug(f'replace [inf,-inf] to {self.replace_inf_values}')
            X = X.replace([np.inf, -np.inf], self.replace_inf_values)

        logger.debug('collect meta info from data')
        df_meta = {}
        for col_info in zip(X.columns.to_list(), X.dtypes):
            dtype = str(col_info[1])
            if df_meta.get(dtype) is None:
                df_meta[dtype] = []
            df_meta[dtype].append(col_info[0])
        self.df_meta_ = df_meta
        logger.info(f'dataframe meta:{self.df_meta_}')
        self.columns_ = X.columns.to_list()
        return X, y

    def transform(self, X, y=None, copy_data=True):
        if copy_data:
            X = copy.deepcopy(X)
            if y is not None:
                y = copy.deepcopy(y)
        X, y = self.clean_data(X, y)
        if self.df_meta_ is not None:
            logger.debug('processing with meta info')
            all_cols = []
            for dtype, cols in self.df_meta_.items():
                all_cols += cols
                X[cols] = X[cols].astype(dtype)
            drop_cols = set(X.columns.to_list()) - set(all_cols)
            X = X[all_cols]
            logger.debug(f'droped columns:{drop_cols}')

        if self.replace_inf_values is not None:
            logger.debug(f'replace [inf,-inf] to {self.replace_inf_values}')
            X = X.replace([np.inf, -np.inf], self.replace_inf_values)

        X = X[self.columns_]
        if y is None:
            return X
        else:
            return X, y

    def append_drop_columns(self, columns):
        if self.df_meta_ is None:
            if self.drop_columns is None:
                self.drop_columns = []
            self.drop_columns = list(set(self.drop_columns + columns))
        else:
            meta = {}
            for dtype, cols in self.df_meta_.items():
                meta[dtype] = [c for c in cols if c not in columns]
            self.df_meta_ = meta

        self.columns_ = [c for c in self.columns_ if c not in columns]

    def _repr_html_(self):
        cleaner_info = []
        cleaner_info.append(('Meta', self.df_meta_))
        cleaner_info.append(('Dropped constant columns', self.dropped_constant_columns_))
        cleaner_info.append(('Dropped idness columns', self.dropped_idness_columns_))
        cleaner_info.append(('Dropped duplicated columns', self.dropped_duplicated_columns_))
        cleaner_info.append(('-------------params-------------', '-------------values-------------'))
        cleaner_info.append(('nan_chars', self.nan_chars))
        cleaner_info.append(('correct_object_dtype', self.correct_object_dtype))
        cleaner_info.append(('drop_constant_columns', self.drop_constant_columns))
        cleaner_info.append(('drop_label_nan_rows', self.drop_label_nan_rows))
        cleaner_info.append(('drop_idness_columns', self.drop_idness_columns))
        cleaner_info.append(('replace_inf_values', self.replace_inf_values))
        cleaner_info.append(('drop_columns', self.drop_columns))
        cleaner_info.append(('drop_duplicated_columns', self.drop_duplicated_columns))
        cleaner_info.append(('reduce_mem_usage', self.reduce_mem_usage))
        cleaner_info.append(('int_convert_to', self.int_convert_to))

        html = pd.DataFrame(cleaner_info, columns=['key', 'value'])._repr_html_()
        return html
