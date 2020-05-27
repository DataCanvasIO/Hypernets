# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from contrib.deeptables.models import *
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.searcher import OptimizeDirection
from hypernets.core.callbacks import SummaryCallback
import pandas as pd


class Test_HyperDT():
    def test_hyper_dt(self):
        rs = RandomSearcher(default_space, optimize_direction=OptimizeDirection.Maximize, )
        hdt = HyperDT(rs,
                      callbacks=[SummaryCallback()],
                      reward_metric='accuracy',
                      max_trails=3,
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
        hdt.search(df, y, df, y)
        assert hdt.best_model
        best_trial = hdt.get_best_trail()

        estimator = hdt.final_train(best_trial.space_sample, df, y)
        score = estimator.predict(df)
        result = estimator.evaluate(df, y)
        assert len(score) == 100
        assert result
        assert isinstance(estimator.model, DeepTable)
