# -*- coding:utf-8 -*-
"""

"""

import cudf
import cupy

from ..data_cleaner import DataCleaner, _CleanerHelper


class CumlDataCleaner(DataCleaner):
    @staticmethod
    def get_helper(X, y):
        if isinstance(X, (cudf.DataFrame, cudf.Series)):
            return _CumlCleanerHelper()
        else:
            return DataCleaner.get_helper(X, y)


class _CumlCleanerHelper(_CleanerHelper):
    @staticmethod
    def _get_duplicated_columns(df):
        columns = df.columns.to_list()
        duplicates = set()

        for i, c in enumerate(columns[:-1]):
            if c in duplicates:
                continue
            for nc in columns[i + 1:]:
                if df[c].equals(df[nc]):
                    duplicates.add(nc)

        return {c: c in duplicates for c in columns}

    @staticmethod
    def _get_df_uniques(df):
        columns = df.columns.to_list()
        uniques = [df[c].nunique() for c in columns]
        return {c: v for c, v in zip(columns, uniques)}

    @staticmethod
    def replace_nan_chars(X: cudf.DataFrame, nan_chars):
        cat_cols = X.select_dtypes(['object', ])
        if cat_cols.shape[1] > 0:
            cat_cols = cat_cols.replace(nan_chars, cupy.nan)
            X[cat_cols.columns] = cat_cols
        return X
