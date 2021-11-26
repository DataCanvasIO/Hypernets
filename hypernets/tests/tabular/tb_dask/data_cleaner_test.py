# -*- coding:utf-8 -*-
"""

"""
from . import if_dask_ready, is_dask_installed
from ..data_cleaner_test import TestDataCleaner

if is_dask_installed:
    import dask.dataframe as dd


@if_dask_ready
class TestDaskDataCleaner(TestDataCleaner):
    @classmethod
    def setup_class(cls):
        TestDataCleaner.setup_class()
        cls.df = dd.from_pandas(TestDataCleaner.df, npartitions=2)
