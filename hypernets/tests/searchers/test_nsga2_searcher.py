import pytest

from hypernets.core import OptimizeDirection
from hypernets.model.objectives import ElapsedObjective, PredictionObjective
from hypernets.searchers.nsga_searcher import NSGAIISearcher, NSGAIndividual

from sklearn.preprocessing import LabelEncoder
from hypernets.core.random_state import set_random_state
from hypernets.examples.plain_model import PlainSearchSpace, PlainModel
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from sklearn.model_selection import train_test_split

set_random_state(1234)

from hypernets.core.callbacks import *


def test_fast_non_dominated_sort():

    i1 = NSGAIndividual("1", np.array([0.1, 0.3]), None)
    i2 = NSGAIndividual("2", np.array([0.2, 0.3]), None)

    l = NSGAIISearcher.fast_non_dominated_sort([i1, i2])
    assert len(l) == 2

    assert l[0][0] == i1
    assert l[1][0] == i2

    # first rank has two element
    i3 = NSGAIndividual("3", np.array([0.3, 0.1]), None)
    l = NSGAIISearcher.fast_non_dominated_sort([i1, i2, i3])
    assert len(l) == 2
    assert i1 in l[0]
    assert i3 in l[0]
    assert l[1][0] == i2

    i4 = NSGAIndividual("4", np.array([0.25, 0.3]), None)
    l = NSGAIISearcher.fast_non_dominated_sort([i1, i2, i3, i4])
    assert len(l) == 3
    assert l[2][0] == i4


def test_crowd_distance_sort():
    i1 = NSGAIndividual("1", np.array([0.10, 0.30]), None)
    i2 = NSGAIndividual("2", np.array([0.11, 0.25]), None)
    i3 = NSGAIndividual("3", np.array([0.12, 0.19]), None)
    i4 = NSGAIndividual("4", np.array([0.13, 0.10]), None)

    pop = NSGAIISearcher.crowding_distance_assignment([i1, i2, i3, i4])  # i1, i2, i3, i4 are in the same rank

    assert i1.distance == i4.distance == float("inf")  # i1 & i4 are always selected
    assert i3.distance > i2.distance  # i3 is more sparsity


@pytest.mark.parametrize('recombination', ["shuffle", "uniform", "single_point"])
def test_nsga2_training(recombination: str):

    df = dsutils.load_bank()
    df['y'] = LabelEncoder().fit_transform(df['y'])

    df.drop(['id'], axis=1, inplace=True)
    X_train, X_test = train_test_split(df, test_size=0.8, random_state=1234)

    y_train = X_train.pop('y')
    y_test = X_test.pop('y')

    search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)
    rs = NSGAIISearcher(search_space, objectives=[ElapsedObjective(),
                                                  PredictionObjective('logloss', OptimizeDirection.Minimize)],
                        recombination=recombination, population_size=3)

    hk = PlainModel(rs, task='binary', callbacks=[SummaryCallback()], transformer=MultiLabelEncoder)

    hk.search(X_train, y_train, X_test, y_test, max_trials=5)

    len(hk.history.trials)
    assert hk.get_best_trial()


def test_non_consistent_direction():

    df = dsutils.load_bank()
    df['y'] = LabelEncoder().fit_transform(df['y'])

    df.drop(['id'], axis=1, inplace=True)
    X_train, X_test = train_test_split(df, test_size=0.8, random_state=1234)

    y_train = X_train.pop('y')
    y_test = X_test.pop('y')

    search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)
    rs = NSGAIISearcher(search_space, objectives=[ElapsedObjective(),
                                                  PredictionObjective('auc', OptimizeDirection.Maximize)],
                        recombination='single_point', population_size=10)

    hk = PlainModel(rs, task='binary', callbacks=[SummaryCallback()], transformer=MultiLabelEncoder)

    hk.search(X_train, y_train, X_test, y_test, max_trials=30)

    len(hk.history.trials)
    assert hk.get_best_trial()

    ns = rs.get_nondominated_set()
    print(ns)

    rs.plot_pf(consistent_direction=False)


