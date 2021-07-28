# -*- coding:utf-8 -*-
"""

"""
import copy
from functools import partial

import dask
import numpy as np
import pandas as pd
from dask import dataframe as dd

from hypernets.tabular.cfg import TabularCfg as cfg
from hypernets.tabular.column_selector import column_object, column_int, column_object_category_bool_int, \
    AutoCategoryColumnSelector
from hypernets.utils import logging

logger = logging.get_logger(__name__)


def _reduce_mem_usage(df, excludes=None):
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
        if excludes is not None and col in excludes:
            continue
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
    if logger.is_info_enabled():
        logger.info('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'
                    .format(end_mem, 100 * (start_mem - end_mem) / start_mem))


def _drop_duplicated_columns(X, excludes=None):
    if isinstance(X, dd.DataFrame):
        duplicates = X.reduction(chunk=lambda c: pd.DataFrame(c.T.duplicated()).T,
                                 aggregate=lambda a: np.all(a, axis=0)).compute()
    else:
        duplicates = X.T.duplicated()

    dup_cols = [i for i, v in duplicates.items() if v and (excludes is None or i not in excludes)]
    columns = [c for c in X.columns.to_list() if c not in dup_cols]
    X = X[columns]
    return X, dup_cols


def _detect_dtype(dtype, df):
    result = {}
    df = df.copy()
    for col in df.columns.to_list():
        try:
            df[col] = df[col].astype(dtype)
            result[col] = [True]  # as-able
        except:
            result[col] = [False]
    return pd.DataFrame(result)


def _correct_object_dtype_as(X, df_meta):
    for dtype, columns in df_meta.items():
        columns = [c for c in columns if str(X[c].dtype) != dtype]
        if len(columns) == 0:
            continue

        if isinstance(X, dd.DataFrame):
            correctable = X[columns].reduction(chunk=partial(_detect_dtype, dtype),
                                               aggregate=lambda a: np.all(a, axis=0),
                                               meta={c: 'bool' for c in columns}).compute()
            correctable = [i for i, v in correctable.items() if v]
            # for col in correctable:
            #     X[col] = X[col].astype(dtype)
            if correctable:
                X[correctable] = X[correctable].astype(dtype)
            logger.info(f'Correct columns [{",".join(correctable)}] to {dtype}.')
        else:
            if dtype in ['object', 'str']:
                X[columns] = X[columns].astype(dtype)
            else:
                for col in columns:
                    try:
                        if str(X[col].dtype) != str(dtype):
                            X[col] = X[col].astype(dtype)
                    except Exception as e:
                        if logger.is_debug_enabled():
                            logger.debug(f'Correct object column [{col}] as {dtype} failed. {e}')

    return X


def _correct_object_dtype(X, df_meta=None, excludes=None):
    if df_meta is None:
        Xt = X[[c for c in X.columns.to_list() if c not in excludes]] if excludes else X
        cat_exponent = cfg.auto_categorize_shape_exponent
        cats = AutoCategoryColumnSelector(cat_exponent=cat_exponent)(Xt) if cfg.auto_categorize else []
        if logger.is_info_enabled() and len(cats) > 0:
            auto_cats = list(filter(lambda _: str(X[_].dtype) != 'object', cats))
            if auto_cats:
                logger.info(f'auto categorize columns: {auto_cats}')

        cons = [c for c in column_object(Xt) if c not in cats]
        df_meta = {'object': cats, 'float': cons}
    X = _correct_object_dtype_as(X, df_meta)

    return X


def _drop_constant_columns(X, excludes=None):
    if isinstance(X, dd.DataFrame):
        nunique = X.reduction(chunk=lambda c: pd.DataFrame(c.nunique(dropna=True)).T,
                              aggregate=np.max).compute()
    else:
        nunique = X.nunique(dropna=True)

    const_cols = [i for i, v in nunique.items() if v <= 1 and (excludes is None or i not in excludes)]
    columns = [c for c in X.columns.to_list() if c not in const_cols]
    X = X[columns]
    return X, const_cols


def _drop_idness_columns(X, excludes=None):
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

    threshold = cfg.idness_threshold
    dropped = [i for i, v in nunique.items() if v / rows > threshold and (excludes is None or i not in excludes)]
    columns = [c for c in X.columns.to_list() if c not in dropped]
    X = X[columns]
    return X, dropped


class DataCleaner:
    def __init__(self, nan_chars=None, correct_object_dtype=True, drop_constant_columns=True,
                 drop_duplicated_columns=False, drop_label_nan_rows=True, drop_idness_columns=True,
                 replace_inf_values=np.nan, drop_columns=None, reserve_columns=None,
                 reduce_mem_usage=False, int_convert_to='float'):
        self.nan_chars = nan_chars
        self.correct_object_dtype = correct_object_dtype
        self.drop_constant_columns = drop_constant_columns
        self.drop_label_nan_rows = drop_label_nan_rows
        self.drop_idness_columns = drop_idness_columns
        self.replace_inf_values = replace_inf_values
        self.drop_columns = drop_columns
        self.reserve_columns = reserve_columns
        self.drop_duplicated_columns = drop_duplicated_columns
        self.reduce_mem_usage = reduce_mem_usage
        self.int_convert_to = int_convert_to

        # fitted
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
            'reserve_columns': self.reserve_columns,
            'drop_duplicated_columns': self.drop_duplicated_columns,
            'reduce_mem_usage': self.reduce_mem_usage,
            'int_convert_to': self.int_convert_to
        }

    @staticmethod
    def _drop_columns(X, cols):
        if cols is None or len(cols) <= 0:
            return X
        X = X[[c for c in X.columns.to_list() if c not in cols]]
        return X

    def clean_data(self, X, y, df_meta=None):
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

        if self.drop_columns is not None:
            logger.debug(f'drop columns:{self.drop_columns}')
            X = self._drop_columns(X, self.drop_columns)

        if self.drop_duplicated_columns:
            logger.debug('drop duplicated columns')
            if self.dropped_duplicated_columns_ is not None:
                X = self._drop_columns(X, self.dropped_duplicated_columns_)
            else:
                X, self.dropped_duplicated_columns_ = _drop_duplicated_columns(X, self.reserve_columns)

        if self.drop_idness_columns:
            logger.debug('drop idness columns')
            if self.dropped_idness_columns_ is not None:
                X = self._drop_columns(X, self.dropped_idness_columns_)
            else:
                X, self.dropped_idness_columns_ = _drop_idness_columns(X, self.reserve_columns)

        if self.drop_constant_columns:
            logger.debug('drop constant columns')
            if self.dropped_constant_columns_ is not None:
                X = self._drop_columns(X, self.dropped_constant_columns_)
            else:
                X, self.dropped_constant_columns_ = _drop_constant_columns(X, self.reserve_columns)

        if self.correct_object_dtype:
            logger.debug('correct data type for object columns.')
            # for col in column_object(X):
            #     try:
            #         X[col] = X[col].astype('float')
            #     except Exception as e:
            #         logger.error(f'Correct object column [{col}] failed. {e}')
            X = _correct_object_dtype(X, df_meta, excludes=self.reserve_columns)

        if self.int_convert_to is not None:
            logger.debug(f'convert int type to {self.int_convert_to}')
            int_cols = column_int(X)
            if self.reserve_columns:
                int_cols = list(filter(lambda _: _ not in self.reserve_columns, int_cols))
            X[int_cols] = X[int_cols].astype(self.int_convert_to)

        if self.replace_inf_values is not None:
            logger.debug(f'replace [inf,-inf] to {self.replace_inf_values}')
            X = X.replace([np.inf, -np.inf], self.replace_inf_values)

        o_cols = column_object(X)
        if self.reserve_columns:
            o_cols = list(filter(lambda _: _ not in self.reserve_columns, o_cols))
        if o_cols:
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
            _reduce_mem_usage(X, excludes=self.reserve_columns)

        logger.debug('collect meta info from data')
        df_meta = {}
        for col_info in zip(X.columns.to_list(), X.dtypes):
            dtype = str(col_info[1])
            if df_meta.get(dtype) is None:
                df_meta[dtype] = []
            df_meta[dtype].append(col_info[0])

        logger.info(f'dataframe meta:{df_meta}')
        self.df_meta_ = df_meta
        self.columns_ = X.columns.to_list()

        return X, y

    def transform(self, X, y=None, copy_data=True):
        if copy_data:
            X = copy.deepcopy(X)
            if y is not None:
                y = copy.deepcopy(y)
        orig_columns = X.columns.to_list()
        X, y = self.clean_data(X, y, df_meta=self.df_meta_)
        # if self.df_meta_ is not None:
        #     logger.debug('processing with meta info')
        #     all_cols = []
        #     for dtype, cols in self.df_meta_.items():
        #         all_cols += cols
        #         X[cols] = X[cols].astype(dtype)
        #     drop_cols = set(X.columns.to_list()) - set(all_cols)
        #     X = X[all_cols]
        #     logger.debug(f'droped columns:{drop_cols}')

        X = X[self.columns_]
        if logger.is_info_enabled():
            dropped = [c for c in orig_columns if c not in self.columns_]
            logger.info(f'drop columns: {dropped}')

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
        cleaner_info.append(('reserve_columns', self.reserve_columns))
        cleaner_info.append(('drop_duplicated_columns', self.drop_duplicated_columns))
        cleaner_info.append(('reduce_mem_usage', self.reduce_mem_usage))
        cleaner_info.append(('int_convert_to', self.int_convert_to))

        html = pd.DataFrame(cleaner_info, columns=['key', 'value'])._repr_html_()
        return html
