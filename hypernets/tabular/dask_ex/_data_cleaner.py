# -*- coding:utf-8 -*-
"""

"""
from functools import partial

import numpy as np
import pandas as pd
from dask import dataframe as dd, array as da

from hypernets.utils import logging
from ..data_cleaner import DataCleaner, _CleanerHelper

logger = logging.get_logger(__name__)


class DaskDataCleaner(DataCleaner):
    @staticmethod
    def get_helper(X, y):
        if isinstance(X, (dd.DataFrame, dd.Series, da.Array)):
            return _DaskCleanerHelper()
        else:
            return DataCleaner.get_helper(X, y)


class _DaskCleanerHelper(_CleanerHelper):
    @staticmethod
    def reduce_mem_usage(df, excludes=None):
        raise NotImplementedError('"reduce_mem_usage" is not supported for Dask DataFrame.')

    @staticmethod
    def _get_duplicated_columns(df):
        duplicates = df.reduction(chunk=lambda c: pd.DataFrame(c.T.duplicated()).T,
                                  aggregate=lambda a: np.all(a, axis=0)).compute()
        return duplicates

    @staticmethod
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

    def _correct_object_dtype_as(self, X, df_meta):
        for dtype, columns in df_meta.items():
            columns = [c for c in columns if str(X[c].dtype) != dtype]
            if len(columns) == 0:
                continue

            correctable = X[columns].reduction(chunk=partial(self._detect_dtype, dtype),
                                               aggregate=lambda a: np.all(a, axis=0),
                                               meta={c: 'bool' for c in columns}).compute()
            correctable = [i for i, v in correctable.items() if v]
            # for col in correctable:
            #     X[col] = X[col].astype(dtype)
            if correctable:
                X[correctable] = X[correctable].astype(dtype)
            logger.info(f'Correct columns [{",".join(correctable)}] to {dtype}.')

        return X
