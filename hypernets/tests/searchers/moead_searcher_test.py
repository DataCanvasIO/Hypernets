import time

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


class TestMOEADSearcher:

    @pytest.mark.parametrize('decomposition', ['pbi',  'weighted_sum', 'tchebicheff'])
    @pytest.mark.parametrize('recombination', ["shuffle", "uniform", "single_point"])
    def test_moead_training(self, decomposition: str, recombination: str):
        t1 = time.time()

        X_train, y_train, X_test, y_test = self.data

        search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)

        objectives = [ElapsedObjective(),
                      PredictionObjective('logloss', OptimizeDirection.Minimize)]

        rs = MOEADSearcher(search_space, objectives=objectives,
                           decomposition=decomposition, recombination=recombination, n_sampling=2)

        hk = PlainModel(rs, task='binary', callbacks=[SummaryCallback()], transformer=MultiLabelEncoder)
        # N => C_3^1
        assert rs.population_size == 3
        hk.search(X_train, y_train, X_test, y_test, max_trials=8)

        len(hk.history.trials)
        assert hk.get_best_trial()
        rs.plot_pf()
        print(time.time() - t1)

    @classmethod
    def setup_class(cls):
        df = dsutils.load_bank().sample(1000)
        df['y'] = LabelEncoder().fit_transform(df['y'])

        df.drop(['id'], axis=1, inplace=True)
        X_train, X_test = train_test_split(df, test_size=0.8, random_state=1234)

        y_train = X_train.pop('y')
        y_test = X_test.pop('y')

        cls.data = (X_train, y_train, X_test, y_test)

    @classmethod
    def teardown_class(cls):
        del cls.data



if __name__ == '__main__':
    # test_moead_training("tchebicheff", "shuffle")
    # test_moead_training("tchebicheff", "single_point")
    # test_moead_training("tchebicheff", "shuffle")
    tm = TestMOEADSearcher()
    tm.setup_class()
    tm.test_moead_training("weighted_sum", "shuffle")
    tm.teardown_class()


