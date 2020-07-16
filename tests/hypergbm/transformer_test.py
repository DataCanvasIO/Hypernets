# -*- coding:utf-8 -*-
"""

"""
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
# from sklearn_pandas import DataFrameMapper
from hypernets.frameworks.ml.sklearn_pandas import DataFrameMapper
import numpy as np
import pandas as pd
from hypernets.frameworks.ml.column_selector import *


def get_df():
    X = pd.DataFrame(
        {
            "a": ['a', 'b', np.nan],
            "b": list(range(1, 4)),
            "c": np.arange(3, 6).astype("u1"),
            "d": np.arange(4.0, 7.0, dtype="float64"),
            "e": [True, False, True],
            "f": pd.Categorical(['c', 'd', np.nan]),
            "g": pd.date_range("20130101", periods=3),
            "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
            "i": pd.date_range("20130101", periods=3, tz="CET"),
            "j": pd.period_range("2013-01", periods=3, freq="M"),
            "k": pd.timedelta_range("1 day", periods=3),
            "l": [1, 10, 1000]
        }
    )
    y = [1, 1, 0]
    return X, y


class Test_Transformer():
    def test_pca(self):
        ct = make_column_transformer(
            (PCA(2), column_number_exclude_timedelta)
        )

        X, y = get_df()
        x_new = ct.fit_transform(X, y)
        assert x_new.shape == (3, 2)

        dfm = DataFrameMapper(
            [(column_number_exclude_timedelta, PCA(2)),
             (column_object_category_bool, [SimpleImputer(strategy='constant'), OneHotEncoder()]),
             (column_number_exclude_timedelta, PolynomialFeatures(2)),
             ], input_df=True, df_out=True
        )
        x_new = dfm.fit_transform(X, y)
        assert x_new.columns.to_list() == ['b_c_d_l_0', 'b_c_d_l_1', 'a_a', 'a_b', 'a_missing_value', 'e_False',
                                           'e_True', 'f_c', 'f_d', 'f_missing_value', '1', 'b', 'c', 'd', 'l',
                                           'b^2', 'b c', 'b d', 'b l', 'c^2', 'c d', 'c l', 'd^2', 'd l', 'l^2']
