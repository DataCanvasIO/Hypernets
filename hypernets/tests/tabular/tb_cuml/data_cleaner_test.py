# -*- coding:utf-8 -*-
"""

"""
from . import if_cuml_ready, is_cuml_installed
from ..data_cleaner_test import TestDataCleaner as _TestDataCleaner

if is_cuml_installed:
    import cudf


@if_cuml_ready
class TestCumlDataCleaner(_TestDataCleaner):
    @staticmethod
    def load_data():
        return cudf.from_pandas(_TestDataCleaner.load_data())
