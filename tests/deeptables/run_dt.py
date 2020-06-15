# -*- coding:utf-8 -*-
"""

"""
import sys

sys.path.append('../../../Hypernets-incubator')

from contrib.deeptables.models import *
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.searcher import OptimizeDirection
from hypernets.core.callbacks import SummaryCallback, FileLoggingCallback
from hypernets.searchers.mcts_searcher import MCTSSearcher
from hypernets.searchers.evolution_searcher import EvolutionSearcher
from hypernets.core.trial import DiskTrailStore
from deeptables.datasets import dsutils
from sklearn.model_selection import train_test_split

disk_trail_store = DiskTrailStore('~/jack/trail_store')

#searcher = MCTSSearcher(mini_dt_space, max_node_space=0,
#                        optimize_direction=OptimizeDirection.Maximize,
#                        dataset_id='(26048, 14)', trail_store=disk_trail_store)
# searcher = RandomSearcher(mini_dt_space, optimize_direction=OptimizeDirection.Maximize ,dataset_id='(26048, 14)', trail_store=disk_trail_store)
searcher = EvolutionSearcher(mini_dt_space, 200, 100, regularized=True, candidates_size=30, dataset_id='(26048, 14)', optimize_direction=OptimizeDirection.Maximize, trail_store=disk_trail_store)

hdt = HyperDT(searcher,
              callbacks=[SummaryCallback(), FileLoggingCallback(searcher)],
              reward_metric='AUC',
              max_trails=2000,
              # dnn_params={
              #     'dnn_units': ((256, 0, False), (256, 0, False)),
              #     'dnn_activation': 'relu',
              # },
              earlystopping_patience=1,
              trail_store=disk_trail_store
              )

df = dsutils.load_adult()
# df.drop(['id'], axis=1, inplace=True)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
X = df_train
y = df_train.pop(14)
y_test = df_test.pop(14)
# dataset_id='adult_whole_data',
hdt.search(df_train, y, df_test, y_test, batch_size=256, epochs=10, verbose=1, )
assert hdt.best_model
best_trial = hdt.get_best_trail()

estimator = hdt.final_train(best_trial.space_sample, df_train, y)
score = estimator.predict(df_test)
result = estimator.evaluate(df_test, y_test)
print(result)
