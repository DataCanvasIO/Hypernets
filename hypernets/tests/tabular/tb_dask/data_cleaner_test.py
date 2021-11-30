# -*- coding:utf-8 -*-
"""

"""
from . import if_dask_ready, is_dask_installed
from ..data_cleaner_test import TestDataCleaner as _TestDataCleaner

if is_dask_installed:
    import dask.dataframe as dd


@if_dask_ready
class TestDaskDataCleaner(_TestDataCleaner):
    @staticmethod
    def load_data():
        df = _TestDataCleaner.load_data()
        return dd.from_pandas(df, npartitions=2)
