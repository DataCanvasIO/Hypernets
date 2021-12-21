# -*- coding:utf-8 -*-
"""

"""

import cudf
import cupy

from ._transformer import Localizable, copy_attrs_as_local
from ..data_cleaner import DataCleaner, _CleanerHelper


class CumlDataCleaner(DataCleaner, Localizable):
    @staticmethod
    def get_helper(X, y):
        if isinstance(X, (cudf.DataFrame, cudf.Series)):
            return _CumlCleanerHelper()
        else:
            return DataCleaner.get_helper(X, y)

    def as_local(self):
        target = DataCleaner(nan_chars=self.nan_chars, correct_object_dtype=self.correct_object_dtype,
                             drop_constant_columns=self.drop_constant_columns,
                             drop_duplicated_columns=self.drop_duplicated_columns,
                             drop_label_nan_rows=self.drop_label_nan_rows,
                             drop_idness_columns=self.drop_idness_columns,
                             replace_inf_values=self.replace_inf_values,
                             drop_columns=self.drop_columns,
                             reserve_columns=self.reserve_columns,
                             reduce_mem_usage=self.reduce_mem_usage,
                             int_convert_to=self.int_convert_to)
        copy_attrs_as_local(self, target, 'df_meta_', 'columns_', 'dropped_constant_columns_',
                   'dropped_idness_columns_', 'dropped_duplicated_columns_')

        return target


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
    def replace_nan_chars(X: cudf.DataFrame, nan_chars):
        cat_cols = X.select_dtypes(['object', ])
        if cat_cols.shape[1] > 0:
            cat_cols = cat_cols.replace(nan_chars, cupy.nan)
            X[cat_cols.columns] = cat_cols
        return X
