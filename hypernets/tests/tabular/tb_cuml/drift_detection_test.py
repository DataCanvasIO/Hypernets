# -*- coding:utf-8 -*-
"""

"""
import pandas as pd
from pandas.util import hash_pandas_object

from hypernets.tabular.datasets.dsutils import load_bank
from . import if_cuml_ready, is_cuml_installed

if is_cuml_installed:
    import cudf
    from hypernets.tabular.cuml_ex import CumlToolBox

    dd_selector = CumlToolBox.feature_selector_with_drift_detection


@if_cuml_ready
class Test_drift_detection:
    def test_shift_score(self):
        df = load_bank().head(1000)
        df = cudf.from_pandas(df)
        selector = dd_selector()
        scores = selector._covariate_shift_score(df[:700], df[700:])
        print('_covariate_shift_score', scores)
        assert scores['id'] >=0.95

    def test_feature_selection(self):
        df = load_bank()
        df = cudf.from_pandas(df)
        y = df.pop('y')
        p = int(df.shape[0] * 0.8)
        X_train = df[:p]
        X_test = df[p:]
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
        df = cudf.from_pandas(load_bank())
        y = df.pop('y')
        X_train, X_test = CumlToolBox.train_test_split(df.copy(), train_size=0.7, shuffle=True, random_state=9527)
        dd = dd_selector().get_detector()
        dd.fit(X_train, X_test)

        assert len(dd.feature_names_) == 17
        assert len(dd.feature_importances_) == 17
        assert dd.auc_
        assert len(dd.estimator_) == 5

        proba = dd.predict_proba(df)
        assert proba.shape[0] == df.shape[0]

        df = cudf.from_pandas(load_bank())
        y = df.pop('y')
        p = int(df.shape[0] * 0.2)
        X_train, X_test, y_train, y_test = dd.train_test_split(df.copy(), y, test_size=0.2)
        assert X_train.shape == (df.shape[0] - p, df.shape[1])
        assert y_train.shape == (df.shape[0] - p,)
        assert X_test.shape == (p, df.shape[1])
        assert y_test.shape == (p,)

        df['y'] = y
        X_train['y'] = y_train
        X_test['y'] = y_test
        df, X_train, X_test = CumlToolBox.to_local(df, X_train, X_test)
        df_split = pd.concat([X_train, X_test])
        df_hash = hash_pandas_object(df).sort_values()
        splitted_hash = hash_pandas_object(df_split).sort_values()
        assert (df_hash == splitted_hash).all()
