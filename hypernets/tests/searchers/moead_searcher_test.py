import os.path
from pathlib import Path
import sys

from hypernets.core.random_state import set_random_state
from hypernets.examples.plain_model import PlainSearchSpace, PlainModel
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from sklearn.model_selection import train_test_split

set_random_state(1234)

from hypernets.core.callbacks import *
from hypernets.searchers.moead_searcher import MOEADSearcher


def test_moead_training():

    params = {
        'decomposition': 'pbi',
        'n_sampling': 5,
        'pbi_theta': 1,
    }

    p_decomposition = params['decomposition']
    p_n_sampling = int(params['n_sampling'])

    df = dsutils.load_bank()
    df.drop(['id'], axis=1, inplace=True)
    X_train, X_test = train_test_split(df, test_size=0.8, random_state=1234)

    y_train = X_train.pop('y')
    y_test = X_test.pop('y')

    search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)
    objectives = ['logloss', 'elapsed']

    rs = MOEADSearcher(search_space, n_objectives=len(objectives),
                       decomposition=p_decomposition, n_sampling=2)

    hk = PlainModel(rs, task='binary', reward_metric=objectives,
                    callbacks=[SummaryCallback()], transformer=MultiLabelEncoder)

    hk.search(X_train, y_train, X_test, y_test, max_trials=20)

    len(hk.history.trials)
    assert hk.get_best_trial()


if __name__ == '__main__':
    test_moead_training()
