import os.path
from pathlib import Path
import sys

from sklearn.preprocessing import LabelEncoder

from hypernets.core import OptimizeDirection
from hypernets.core.random_state import set_random_state
from hypernets.examples.plain_model import PlainSearchSpace, PlainModel
from hypernets.model.objectives import PredictionObjective, ElapsedObjective
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from sklearn.model_selection import train_test_split

set_random_state(1234)

from hypernets.core.callbacks import *
from hypernets.searchers.moead_searcher import MOEADSearcher

import pytest


@pytest.mark.parametrize('decomposition', ['pbi',  'weighted_sum', 'tchebicheff'])
# @pytest.mark.parametrize('decomposition', ['tchebicheff'])
@pytest.mark.parametrize('recombination', ["shuffle", "uniform", "single_point"])
def test_moead_training(decomposition: str, recombination: str):

    df = dsutils.load_bank()
    df['y'] = LabelEncoder().fit_transform(df['y'])

    df.drop(['id'], axis=1, inplace=True)
    X_train, X_test = train_test_split(df, test_size=0.8, random_state=1234)

    y_train = X_train.pop('y')
    y_test = X_test.pop('y')

    search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)

    objectives = (ElapsedObjective(),
                  PredictionObjective('logloss', OptimizeDirection.Minimize))

    rs = MOEADSearcher(search_space, objectives=objectives,
                       decomposition=decomposition, recombination=recombination, n_sampling=2)

    hk = PlainModel(rs, task='binary', callbacks=[SummaryCallback()], transformer=MultiLabelEncoder)

    hk.search(X_train, y_train, X_test, y_test, max_trials=10)

    len(hk.history.trials)
    assert hk.get_best_trial()


if __name__ == '__main__':
    # test_moead_training("tchebicheff", "shuffle")
    # test_moead_training("tchebicheff", "single_point")
    test_moead_training("tchebicheff", "uniform")
