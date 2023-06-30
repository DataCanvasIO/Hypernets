# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import pytest

from . import if_dask_ready, is_dask_installed
from ..feature_importance_test import TestPermutationImportance as _TestPermutationImportance

if is_dask_installed:
    import dask.dataframe as dd


@if_dask_ready
@pytest.mark.xfail(reasone='to be fixed')
class TestDaskPermutationImportance(_TestPermutationImportance):
    @staticmethod
    def load_data():
        df = _TestPermutationImportance.load_data()
        df = dd.from_pandas(df, npartitions=2)
        return df
