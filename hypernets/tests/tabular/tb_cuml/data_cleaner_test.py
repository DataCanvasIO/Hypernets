# -*- coding:utf-8 -*-
"""

"""
from . import if_cuml_ready, is_cuml_installed
from ..data_cleaner_test import TestDataCleaner

if is_cuml_installed:
    import cudf


@if_cuml_ready
class TestDaskDataCleaner(TestDataCleaner):
    @classmethod
    def setup_class(cls):
        TestDataCleaner.setup_class()
        cls.df = cudf.from_pandas(TestDataCleaner.df)
