# -*- coding:utf-8 -*-
"""

"""
from deeptables.datasets import dsutils
from pandas import DataFrame
import pandas as pd
from hypernets.frameworks.ml.transformers import ColumnTransformer, DataFrameMapper
from hypernets.frameworks.ml.common_ops import categorical_pipeline_simple, numeric_pipeline
from hypernets.frameworks.ml.estimators import LightGBMEstimator
from hypernets.frameworks.ml.hyper_gbm import HyperGBMEstimator
from hypernets.core.ops import *

from .common_ops_test import get_space_categorical_pipeline, get_space_num_cat_pipeline_complex
from sklearn.model_selection import train_test_split


def get_space_multi_dataframemapper(default=False):
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = numeric_pipeline(seq_no=0)(input)
        p2 = categorical_pipeline_simple(seq_no=0)(input)
        p3 = DataFrameMapper(default=default, input_df=True, df_out=True)([p1, p2])  # passthrough

        p4 = numeric_pipeline(seq_no=1)(p3)
        p5 = categorical_pipeline_simple(seq_no=1)(p3)
        p6 = DataFrameMapper(default=default, input_df=True, df_out=True)([p4, p5])
        est = LightGBMEstimator(task='binary', fit_kwargs={})(p6)
        space.set_inputs(input)
    return space


def get_df():
    X = DataFrame(
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


class Test_Estimator():
    def test_build_pipeline(self):
        space = get_space_categorical_pipeline()
        space.random_sample()
        estimator = HyperGBMEstimator('binary', space)
        X, y = get_df()
        df_1 = estimator.pipeline.fit_transform(X, y)
        assert list(df_1.columns) == ['a', 'e', 'f']
        assert df_1.shape == (3, 3)

        space = get_space_multi_dataframemapper()
        space.random_sample()
        estimator = HyperGBMEstimator('binary', space)
        X, y = get_df()
        df_1 = estimator.pipeline.fit_transform(X, y)
        assert list(df_1.columns) == ['a', 'e', 'f', 'b', 'c', 'd', 'l']
        assert df_1.shape == (3, 7)

    def test_bankdata(self):
        space = get_space_num_cat_pipeline_complex()
        space.assign_by_vectors([1, 1, 0, 1, 1])
        estimator = HyperGBMEstimator('binary', space)
        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)
        X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        estimator.fit(X_train, y_train)
        scores = estimator.evaluate(X_test, y_test, metrics=['accuracy'])
        assert scores
