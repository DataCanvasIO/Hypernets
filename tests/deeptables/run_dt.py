# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from contrib.deeptables.models import *
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.searcher import OptimizeDirection
from hypernets.core.callbacks import SummaryCallback, FileLoggingCallback
from hypernets.searchers.mcts_searcher import MCTSSearcher
from hypernets.searchers.evolution_searcher import EvolutionSearcher

from deeptables.datasets import dsutils
from sklearn.model_selection import train_test_split

#searcher = MCTSSearcher(default_space, max_node_space=10, optimize_direction=OptimizeDirection.Maximize)
#searcher = RandomSearcher(default_space, optimize_direction=OptimizeDirection.Maximize )
searcher = EvolutionSearcher(default_space, 50, 30, regularized=True, optimize_direction=OptimizeDirection.Maximize)

hdt = HyperDT(searcher,
              callbacks=[SummaryCallback(), FileLoggingCallback(searcher)],
              reward_metric='AUC',
              max_trails=1000,
              dnn_params={
                  'dnn_units': ((256, 0, False), (256, 0, False)),
                  'dnn_activation': 'relu',
              },
              earlystopping_patience=1,
              )

df = dsutils.load_adult()
# df.drop(['id'], axis=1, inplace=True)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
X = df_train
y = df_train.pop(14)
y_test = df_test.pop(14)

hdt.search(df_train, y, df_test, y_test, batch_size=256, epochs=10, verbose=1, )
assert hdt.best_model
best_trial = hdt.get_best_trail()

estimator = hdt.final_train(best_trial.space_sample, df_train, y)
score = estimator.predict(df_test)
result = estimator.evaluate(df_test, y_test)
print(result)
