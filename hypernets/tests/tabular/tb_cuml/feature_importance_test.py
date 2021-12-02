# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from ..feature_importance_test import TestPermutationImportance as _TestPermutationImportance
from . import if_cuml_ready, is_cuml_installed

if is_cuml_installed:
    import cudf


@if_cuml_ready
class TestCumlPermutationImportance(_TestPermutationImportance):
    @staticmethod
    def load_data():
        df = _TestPermutationImportance.load_data()
        df = cudf.from_pandas(df)
        return df
