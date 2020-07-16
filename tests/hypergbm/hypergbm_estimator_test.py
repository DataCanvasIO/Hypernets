# -*- coding:utf-8 -*-
"""

"""

from pandas import DataFrame
import numpy as np
import pandas as pd
from hypernets.frameworks.ml.preprocessing import ColumnTransformer, DataFrameMapper
from hypernets.frameworks.ml.column_selector import *
from hypernets.frameworks.ml.common_ops import categorical_pipeline, numeric_pipeline
from hypernets.frameworks.ml.estimators import LightGBMEstimator
from hypernets.frameworks.ml.hyper_gbm import HyperGBMEstimator
from hypernets.core.search_space import *
from hypernets.core.ops import *


def get_space_categorical_pipeline():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = categorical_pipeline()(input)
        p3 = DataFrameMapper(input_df=True, df_out=True)([p1])  # passthrough
        est = LightGBMEstimator(task='binary', fit_kwargs={})(p3)
        space.set_inputs(input)
    return space


def get_space_numeric_pipeline():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = numeric_pipeline()(input)
        p3 = DataFrameMapper(input_df=True, df_out=True)([p1])  # passthrough
        est = LightGBMEstimator(task='binary', fit_kwargs={})(p3)
        space.set_inputs(input)
    return space


def get_space_num_cat_pipeline(default=False):
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = numeric_pipeline()(input)
        p2 = categorical_pipeline()(input)
        p3 = DataFrameMapper(default=default, input_df=True, df_out=True)([p1, p2])  # passthrough
        est = LightGBMEstimator(task='binary', fit_kwargs={})(p3)
        space.set_inputs(input)
    return space


def get_space_multi_dataframemapper(default=False):
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = numeric_pipeline(seq_no=0)(input)
        p2 = categorical_pipeline(seq_no=0)(input)
        p3 = DataFrameMapper(default=default, input_df=True, df_out=True)([p1, p2])  # passthrough

        p4 = numeric_pipeline(seq_no=1)(p3)
        p5 = categorical_pipeline(seq_no=1)(p3)
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
        estimator = HyperGBMEstimator(space)
        X, y = get_df()
        df_1 = estimator.pipeline.fit_transform(X, y)
        assert list(df_1.columns) == ['a', 'e', 'f']
        assert df_1.shape == (3, 3)

        space = get_space_multi_dataframemapper()
        space.random_sample()
        estimator = HyperGBMEstimator(space)
        X, y = get_df()
        df_1 = estimator.pipeline.fit_transform(X, y)
        assert list(df_1.columns) == ['b', 'c', 'd', 'l', 'a', 'e', 'f']
        assert df_1.shape == (3, 7)
