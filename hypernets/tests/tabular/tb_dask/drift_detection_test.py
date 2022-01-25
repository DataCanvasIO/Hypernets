# -*- coding:utf-8 -*-
"""

"""
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from hypernets.tabular.datasets.dsutils import load_bank
from . import if_dask_ready, is_dask_installed, setup_dask

if is_dask_installed:
    import dask.dataframe as dd
    from hypernets.tabular.dask_ex import DaskToolBox

    dd_selector = DaskToolBox.feature_selector_with_drift_detection


@if_dask_ready
class Test_drift_detection:
    @classmethod
    def setup_class(cls):
        setup_dask(cls)

    def test_shift_score(self):
        df = load_bank().head(1000)
        df = dd.from_pandas(df, npartitions=2)
        selector = dd_selector()
        df_train = DaskToolBox.select_df(df, np.arange(700))
        df_test = DaskToolBox.select_df(df, np.arange(700, 1000))
        scores = selector._covariate_shift_score(df_train, df_test)
        assert scores['id'] > 0.99

    def test_feature_selection(self):
        df = load_bank()
        df = dd.from_pandas(df, npartitions=2)
        y = df.pop('y')
        p = int(len(df) * 0.8)
        X_train = DaskToolBox.select_df(df, np.arange(p))  # df[:p]
        X_test = DaskToolBox.select_df(df, np.arange(p, len(df)))  # df[p:]
        # = train_test_split(df, train_size=0.7,  random_state=9527)
        selector = dd_selector(remove_shift_variable=False,
                               auc_threshold=0.55,
                               min_features=15,
                               remove_size=0.2)
        remain_features, history, scores = selector.select(X_train, X_test, copy_data=True)
        assert len(remain_features) == 15

        selector = dd_selector(remove_shift_variable=True,
                               auc_threshold=0.55,
                               min_features=15,
                               remove_size=0.2)
        remain_features, history, scores = selector.select(X_train, X_test, copy_data=True)

        assert len(remain_features) == 16

    def test_drift_detector_split(self):
        df = dd.from_pandas(load_bank(), npartitions=2)
        y = df.pop('y')
        X_train, X_test = DaskToolBox.train_test_split(df.copy(), train_size=0.7, shuffle=True, random_state=9527)
        ddr = dd_selector().get_detector()
        ddr.fit(X_train, X_test)

        assert len(ddr.feature_names_) == 17
        assert len(ddr.feature_importances_) == 17
        assert ddr.auc_
        assert len(ddr.estimator_) == 5

        proba = ddr.predict_proba(df)
        assert proba.compute().shape[0] == len(df)

        df = dd.from_pandas(load_bank(), npartitions=2)
        y = df.pop('y')
        p = int(len(df) * 0.2)
        X_train, X_test, y_train, y_test = ddr.train_test_split(df.copy(), y, test_size=0.2, remain_for_train=0.)

        df, X_train, X_test, y_train, y_test = DaskToolBox.compute(df, X_train, X_test, y_train, y_test)
        assert X_train.shape == (df.shape[0] - p, df.shape[1])
        assert y_train.shape == (df.shape[0] - p,)
        assert X_test.shape == (p, df.shape[1])
        assert y_test.shape == (p,)

        df['y'] = y
        X_train['y'] = y_train
        X_test['y'] = y_test
        df_split = pd.concat([X_train, X_test])
        df_hash = hash_pandas_object(df).sort_values()
        splitted_hash = hash_pandas_object(df_split).sort_values()
        assert (df_hash == splitted_hash).all()
