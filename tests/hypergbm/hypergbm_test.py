# -*- coding:utf-8 -*-
"""

"""
from hypernets.frameworks.ml.hyper_gbm import HyperGBM
from hypernets.frameworks.ml.common_ops import get_space_num_cat_pipeline_complex
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection
from deeptables.datasets import dsutils
from sklearn.model_selection import train_test_split
from tests import test_output_dir


class Test_HyperGBM():


    def train_bankdata(self, data_partition):
        rs = RandomSearcher(get_space_num_cat_pipeline_complex, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperGBM(rs, task='classification', reward_metric='accuracy',
                      cache_dir=f'{test_output_dir}/hypergbm_cache',
                      callbacks=[SummaryCallback(), FileLoggingCallback(rs, output_dir=f'{test_output_dir}/hyn_logs')])

        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)

        X_train, X_test, y_train, y_test = data_partition()

        hk.search(X_train, y_train, X_test, y_test, max_trails=30)
        assert hk.best_model
        best_trial = hk.get_best_trail()

        estimator = hk.final_train(best_trial.space_sample, X_train, y_train)
        score = estimator.predict(X_test)
        result = estimator.evaluate(X_test, y_test)
        assert len(score) == 200
        assert result


    def test_model(self):
        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)
        X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        def f():
            return X_train, X_test, y_train, y_test

        self.train_bankdata(f)

    def test_no_categorical(self):

        df = dsutils.load_bank()

        df.drop(['id'], axis=1, inplace=True)
        df = df[['age', 'duration', 'previous', 'y']]

        X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        def f():
            return X_train, X_test, y_train, y_test

        self.train_bankdata(f)

    def test_no_continuous(self):

        df = dsutils.load_bank()

        df.drop(['id'], axis=1, inplace=True)
        df = df[['job', 'education', 'loan', 'y']]

        X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        def f():
            return X_train, X_test, y_train, y_test

        self.train_bankdata(f)

    def test_onehot_handle_unknown(self):
        import sklearn, pandas as pd
        from sklearn_pandas import DataFrameMapper
        dfm = DataFrameMapper([(['name'], sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'))], df_out=True)

        df_train = pd.DataFrame(data={"name": ["a", "b", "c"]})
        df_test = pd.DataFrame(data={"name": ["a", "b", "d"]})
        dfm.fit(df_train)
        ret = dfm.transform(df_test)
        assert ret is not None
        assert ret.shape == (3, 3)
