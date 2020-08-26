# -*- coding:utf-8 -*-
"""

"""
from hypernets.frameworks.ml.feature_generators import FeatureToolsTransformer
from deeptables.datasets import dsutils
from pandas import DataFrame
import pandas as pd
from hypernets.core.ops import *

from .common_ops_test import get_space_categorical_pipeline
from sklearn.model_selection import train_test_split
import featuretools as ft
import math
from datetime import datetime


class Test_FeatureGenerator():

    def test_ft_primitives(self):
        tps = ft.primitives.get_transform_primitives()
        assert tps

    # def test_feature_tools_transformer(self):
    #
    #     df = dsutils.load_bank()
    #     df.drop(['id'], axis=1, inplace=True)
    #     X_train, X_test = train_test_split(df.head(10000), test_size=0.2, random_state=42)
    #     y_train = X_train.pop('y')
    #     y_test = X_test.pop('y')
    #
    #     ftt = FeatureToolsTransformer()
    #     ftt.fit(X_train)
    #     x_t = ftt.transform(X_train)
    #
    #     print(x_t)

    def test_infinity_result(self):
        df = pd.DataFrame(data={"x1": [1,2,3], 'x2': [0, 5, 6]})
        ftt = FeatureToolsTransformer(trans_primitives=['add_numeric', 'divide_numeric'])
        ftt.fit(df)
        x_t = ftt.transform(df)
        assert "x1 + x2" in x_t
        assert "x1 / x2" in x_t

        assert not math.isinf(x_t["x1 / x2"][0])

    def test_datetime_derivation(self):

        df = pd.DataFrame(data={"x1": [datetime.now()]})
        ftt = FeatureToolsTransformer(trans_primitives=["year", "month", "week"])
        ftt.fit(df)

        x_t = ftt.transform(df)
        assert "YEAR(x1)" in x_t
        assert "MONTH(x1)" in x_t
        assert "WEEK(x1)" in x_t

    def test_persist(self, tmp_path: str):
        from os import path as P
        tmp_path = P.join(tmp_path, 'fft.pkl')

        df = pd.DataFrame(data={"x1": [datetime.now()]})
        ftt = FeatureToolsTransformer(trans_primitives=["year", "month", "week"])
        ftt.fit(df)
        import pickle

        with open(tmp_path, 'wb') as f:
            pickle.dump(ftt, f)

        with open(tmp_path, 'rb') as f:
            ftt1 = pickle.load(f)

        x_t = ftt1.transform(df)
        assert "YEAR(x1)" in x_t
        assert "MONTH(x1)" in x_t
        assert "WEEK(x1)" in x_t
