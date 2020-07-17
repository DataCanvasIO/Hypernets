# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from contrib.deeptables.models import *
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.searcher import OptimizeDirection
from hypernets.core.callbacks import SummaryCallback, FileLoggingCallback
import pandas as pd

import numpy as np
import pandas as pd
from deeptables.datasets import dsutils
from sklearn.model_selection import train_test_split


class Test_HyperDT():
    def bankdata(self):
        rs = RandomSearcher(mini_dt_space, optimize_direction=OptimizeDirection.Maximize, )
        hdt = HyperDT(rs,
                      callbacks=[SummaryCallback(), FileLoggingCallback(rs)],
                      reward_metric='accuracy',
                      max_trails=3,
                      dnn_params={
                          'dnn_units': ((256, 0, False), (256, 0, False)),
                          'dnn_activation': 'relu',
                      },
                      )

        df = dsutils.load_bank()
        df.drop(['id'], axis=1, inplace=True)
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        y = df_train.pop('y')
        y_test = df_test.pop('y')

        hdt.search(df_train, y, df_test, y_test)
        assert hdt.best_model
        best_trial = hdt.get_best_trail()

        estimator = hdt.final_train(best_trial.space_sample, df_train, y)
        score = estimator.predict(df)
        result = estimator.evaluate(df, y)
        assert len(score) == 100
        assert result
        assert isinstance(estimator.model, DeepTable)

    def test_default_dt_space(self):
        space = default_dt_space()
        space.random_sample()
        assert space.Module_DnnModule_1.param_values['dnn_layers'] == len(
            space.DT_Module.config.dnn_params['dnn_units'])
        assert space.Module_DnnModule_1.param_values['dnn_units'] == space.DT_Module.config.dnn_params['dnn_units'][0][
            0]
        assert space.Module_DnnModule_1.param_values['dnn_dropout'] == \
               space.DT_Module.config.dnn_params['dnn_units'][0][
                   1]
        assert space.Module_DnnModule_1.param_values['use_bn'] == space.DT_Module.config.dnn_params['dnn_units'][0][
            2]

    def test_hyper_dt(self):
        rs = RandomSearcher(mini_dt_space, optimize_direction=OptimizeDirection.Maximize, )
        hdt = HyperDT(rs,
                      callbacks=[SummaryCallback()],
                      reward_metric='accuracy',
                      dnn_params={
                          'dnn_units': ((256, 0, False), (256, 0, False)),
                          'dnn_activation': 'relu',
                      },
                      )
        x1 = np.random.randint(0, 10, size=(100), dtype='int')
        x2 = np.random.randint(0, 2, size=(100)).astype('str')
        x3 = np.random.randint(0, 2, size=(100)).astype('str')
        x4 = np.random.normal(0.0, 1.0, size=(100))

        y = np.random.randint(0, 2, size=(100), dtype='int')
        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})
        hdt.search(df, y, df, y, max_trails=3, )
        assert hdt.best_model
        best_trial = hdt.get_best_trail()

        estimator = hdt.final_train(best_trial.space_sample, df, y)
        score = estimator.predict(df)
        result = estimator.evaluate(df, y)
        assert len(score) == 100
        assert result
        assert isinstance(estimator.model, DeepTable)
