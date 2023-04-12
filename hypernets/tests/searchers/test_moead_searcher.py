import time

from sklearn.preprocessing import LabelEncoder

from hypernets.core import OptimizeDirection
from hypernets.core.random_state import set_random_state, get_random_state
from hypernets.examples.plain_model import PlainSearchSpace, PlainModel
from hypernets.model.objectives import PredictionObjective, ElapsedObjective, NumOfFeatures
from hypernets.searchers.genetic import create_recombination
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from sklearn.model_selection import train_test_split

set_random_state(1234)

from hypernets.core.callbacks import *
from hypernets.searchers.moead_searcher import MOEADSearcher, create_decomposition

import pytest


class TestMOEADSearcher:

    @pytest.mark.parametrize('decomposition', ['pbi',  'weighted_sum', 'tchebicheff'])
    @pytest.mark.parametrize('recombination', ["shuffle", "uniform", "single_point"])
    @pytest.mark.parametrize('cv', [True, False])
    def test_moead_training(self, decomposition: str, recombination: str, cv: bool):
        t1 = time.time()
        random_state = get_random_state()
        X_train, y_train, X_test, y_test = self.data

        search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)

        objectives = [NumOfFeatures(), PredictionObjective.create('logloss')]

        rs = MOEADSearcher(search_space, objectives=objectives,
                           random_state=random_state,
                           decomposition=create_decomposition(decomposition),
                           recombination=create_recombination(recombination, random_state=random_state),
                           n_sampling=2)

        hk = PlainModel(rs, task='binary', callbacks=[SummaryCallback()],
                        transformer=MultiLabelEncoder, reward_metric='logloss')
        # N => C_3^1
        assert rs.population_size == 3
        hk.search(X_train, y_train, X_test, y_test, max_trials=8, cv=cv)

        len(hk.history.trials)
        assert hk.get_best_trial()
        # rs.plot_pf()
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


