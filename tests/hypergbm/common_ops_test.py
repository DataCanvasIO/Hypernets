# -*- coding:utf-8 -*-
"""

"""

from pandas import DataFrame
import numpy as np
import pandas as pd
from hypernets.frameworks.ml.preprocessing import ColumnTransformer
from hypernets.frameworks.ml.column_selector import *
from hypernets.frameworks.ml.common_ops import categorical_pipeline
from hypernets.core.search_space import *
from hypernets.core.ops import *
from sklearn.pipeline import make_pipeline

ids = []


def get_id(m):
    ids.append(m.id)
    return True


def get_space_categorical_pipeline():
    space = HyperSpace()
    with space.as_default():
        input = HyperInput(name='input1')
        p1 = categorical_pipeline()(input)
        p3 = ColumnTransformer(remainder='passthrough')([p1])
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
        }
    )
    y = [1, 1, 0]
    return X, y


class Test_CommonOps():
    def test_categorical_pipeline(self):
        space = get_space_categorical_pipeline()
        space.random_sample()
        space = space.compile_space()
        global ids
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input1', 'ID_Module_Pipeline_1_input', 'ID_categorical_imputer',
                       'ID_categorical_label_encoder', 'ID_Module_Pipeline_1_output', 'Module_ColumnTransformer_1']

        next, (name, p) = space.Module_ColumnTransformer_1.compose()
        X, y = get_df()
        # p = make_pipeline(steps=[('cat', p), ('pass', 'passthrough')])
        # p = make_pipeline(p, 'passthrough')
        df_1 = p.fit_transform(X, y)
        assert df_1
