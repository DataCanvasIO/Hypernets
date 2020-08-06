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

class Test_FeatureGenerator():

    def test_ft_primitives(self):
        tps = ft.primitives.get_transform_primitives()
        assert tps

    def test_feature_tools_transformer(self):

        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)
        X_train, X_test = train_test_split(df.head(10000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        ftt = FeatureToolsTransformer()
        ftt.fit(X_train)
        x_t = ftt.transform(X_train)
        assert x_t

