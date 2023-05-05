import numpy as np
import pytest

from hypernets.model.objectives import ElapsedObjective, \
    PredictionObjective, NumOfFeatures, PredictionPerformanceObjective, create_objective
from hypernets.searchers.nsga_searcher import NSGAIISearcher, _NSGAIndividual, _RankAndCrowdSortSurvival, \
    _RDominanceSurvival, RNSGAIISearcher

from sklearn.preprocessing import LabelEncoder
from hypernets.core.random_state import set_random_state, get_random_state
from hypernets.examples.plain_model import PlainSearchSpace, PlainModel
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from sklearn.model_selection import train_test_split

from hypernets.searchers.genetic import Individual, create_recombination
from hypernets.utils import const


from hypernets.core.callbacks import *


def get_bankdata():
    df = dsutils.load_bank().head(1000)
    df['y'] = LabelEncoder().fit_transform(df['y'])

    df.drop(['id'], axis=1, inplace=True)
    X_train, X_test = train_test_split(df, test_size=0.8, random_state=1234)

    y_train = X_train.pop('y')
    y_test = X_test.pop('y')
    return X_train, y_train, X_test, y_test


class TestRankAndCrowdSortSurvival:

    @classmethod
    def setup_class(cls):
        survival = _RankAndCrowdSortSurvival(directions=['min', 'min'], population_size=10,
                                             random_state=get_random_state())
        cls.survival = survival

    def test_crowd_distance_sort(self):
        survival = self.survival
        i1 = _NSGAIndividual("1", np.array([0.10, 0.30]), None)
        i2 = _NSGAIndividual("2", np.array([0.11, 0.25]), None)
        i3 = _NSGAIndividual("3", np.array([0.12, 0.19]), None)
        i4 = _NSGAIndividual("4", np.array([0.13, 0.10]), None)

        pop = survival.crowding_distance_assignment([i1, i2, i3, i4])  # i1, i2, i3, i4 are in the same rank

        assert i1.distance == i4.distance == float("inf")  # i1 & i4 are always selected
        assert i3.distance > i2.distance  # i3 is more sparsity

    def test_fast_non_dominated_sort(self):
        survival = self.survival
        i1 = _NSGAIndividual("1", np.array([0.1, 0.3]), None)
        i2 = _NSGAIndividual("2", np.array([0.2, 0.3]), None)

        l = survival.fast_non_dominated_sort([i1, i2])
        assert len(l) == 2

        assert l[0][0] == i1
        assert l[1][0] == i2

        # first rank has two element
        i3 = _NSGAIndividual("3", np.array([0.3, 0.1]), None)
        l = survival.fast_non_dominated_sort([i1, i2, i3])
        assert len(l) == 2
        assert i1 in l[0]
        assert i3 in l[0]
        assert l[1][0] == i2

        i4 = _NSGAIndividual("4", np.array([0.25, 0.3]), None)
        l = survival.fast_non_dominated_sort([i1, i2, i3, i4])
        assert len(l) == 3
        assert l[2][0] == i4

    def test_non_dominated(self):
        survival = self.survival
        i1 = Individual("1", np.array([0.1, 0.2]), None)
        i2 = Individual("1", np.array([0.2, 0.1]), None)
        i3 = Individual("1", np.array([0.2, 0.2]), None)
        i4 = Individual("1", np.array([0.3, 0.2]), None)
        i5 = Individual("1", np.array([0.4, 0.4]), None)

        nondominated_set = survival.calc_nondominated_set([i1, i2, i3, i4, i5])
        assert len(nondominated_set) == 2
        assert i1 in nondominated_set
        assert i2 in nondominated_set


class TestNSGA2:

    @pytest.mark.parametrize('recombination', ["shuffle", "uniform", "single_point"])
    @pytest.mark.parametrize('cv', [True, False])
    #@pytest.mark.parametrize('objective', ['feature_usage', 'nf'])
    def test_nsga2_training(self, recombination: str, cv: bool):
        objective = 'nf'
        set_random_state(1234)
        X_train, y_train, X_test, y_test = get_bankdata()
        recombination_ins = create_recombination(recombination, random_state=get_random_state())
        search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)
        rs = NSGAIISearcher(search_space, objectives=[PredictionObjective.create('accuracy'),
                                                      create_objective(objective)],
                            recombination=recombination_ins, population_size=3)

        # the given reward_metric is in order to ensure SOO working, make it's the same as metrics in MOO searcher
        hk = PlainModel(rs, task='binary', callbacks=[SummaryCallback()], transformer=MultiLabelEncoder,
                        reward_metric='logloss')

        hk.search(X_train, y_train, X_test, y_test, max_trials=5, cv=cv)

        len(hk.history.trials)
        assert hk.get_best_trial()

    def test_non_consistent_direction(self):
        X_train, y_train, X_test, y_test = get_bankdata()

        search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)
        rs = NSGAIISearcher(search_space, objectives=[ElapsedObjective(),
                                                      PredictionObjective.create('auc')],
                            recombination='single_point', population_size=5)

        hk = PlainModel(rs, task='binary', callbacks=[SummaryCallback()], transformer=MultiLabelEncoder)

        hk.search(X_train, y_train, X_test, y_test, max_trials=10)

        len(hk.history.trials)
        assert hk.get_best_trial()

        ns = rs.get_nondominated_set()
        assert ns


class TestRNSGA2:

    @pytest.mark.parametrize('recombination', ["shuffle", "uniform", "single_point"])
    @pytest.mark.parametrize('cv', [True, False])
    def test_nsga2_training(self, recombination: str,  cv: bool):
        set_random_state(1234)
        hk1 = self.run_nsga2_training(recombination=const.COMBINATION_SHUFFLE, cv=cv, objective='nf')
        pop1 = hk1.searcher.get_historical_population()
        scores1 = np.asarray([indi.scores for indi in pop1])
        assert scores1.ndim == 2

        # test search process reproduce by setting random_state
        # set_random_state(1234)  # reset random state
        # hk2 = self.run_nsga2_training(recombination=const.COMBINATION_SHUFFLE)
        # pop2 = hk2.searcher.get_historical_population()
        # scores2 = np.asarray([indi.scores for indi in pop2])
        #
        # assert (scores1 == scores2).all()

    # def reproce_nsga2_training(self):
    #     set_random_state(1234)
    #     hk1 = self.run_nsga2_training(recombination=const.COMBINATION_UNIFORM)
    #     pop1 = hk1.searcher.get_historical_population()
    #     scores1 = np.asarray([indi.scores for indi in pop1])
    #
    #     # test search process reproduce by setting random_state
    #     set_random_state(1234)  # reset random state
    #     hk2 = self.run_nsga2_training(recombination=const.COMBINATION_UNIFORM)
    #     pop2 = hk2.searcher.get_historical_population()
    #     scores2 = np.asarray([indi.scores for indi in pop2])
    #
    #     assert (scores1 == scores2).all()

    def run_nsga2_training(self, recombination: str, cv: bool, objective: str):
        random_state = get_random_state()
        X_train, y_train, X_test, y_test = get_bankdata()
        search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)

        rs = RNSGAIISearcher(search_space, objectives=[PredictionObjective.create('logloss'),
                                                       create_objective(objective)],
                             ref_point=[0.5, 0.5],
                             weights=[0.4, 0.6],
                             random_state=random_state,
                             recombination=create_recombination(recombination, random_state=random_state),
                             population_size=3)

        hk = PlainModel(rs, task='binary', callbacks=[SummaryCallback()], reward_metric='logloss',
                        transformer=MultiLabelEncoder)

        hk.search(X_train, y_train, X_test, y_test, max_trials=5, cv=cv)

        len(hk.history.trials)
        assert hk.get_best_trial()
        # ensure reproduce
        assert hk.searcher.random_state == hk.searcher.recombination.random_state
        assert hk.searcher.random_state == random_state
        return hk


class TestRDominanceSurvival:
    @classmethod
    def setup_class(cls):
        # data setting from fig. 3 in the paper
        reference_point = [0.05, 0.05]
        weights = np.array([0.5, 0.5])  # ignore effect of weights

        scores = np.array([[0.1, 0.1], [0.1, 0.15], [0.2, 0.2], [0.3, 0.3]])

        pop = [_NSGAIndividual(str(i), score, None) for i, score in enumerate(scores)]
        cls.pop = pop

        cls.survival = _RDominanceSurvival(directions=['min', 'min'], random_state=get_random_state(),
                                           ref_point=reference_point,
                                           population_size=len(cls.pop),
                                           weights=weights, threshold=0.3)

    def test_dominate(self):
        a, b, c, d = self.pop
        # _dominate = partial(self.survival.dominate, pop=self.pop)

        def _dominate(x1, x2):
            return self.survival.dominate(ind1=x1, ind2=x2, pop=self.pop)

        assert _dominate(a, b)
        assert _dominate(a, c)
        assert _dominate(a, d)

        assert _dominate(b, c)
        assert _dominate(b, d)

        assert _dominate(b, d)

        assert not _dominate(b, a)
        assert not _dominate(c, a)
        assert not _dominate(d, a)

        assert not _dominate(c, b)
        assert not _dominate(d, b)

        assert not _dominate(d, b)


def test_r_dominate():
    reference_point = [0.2, 0.4]

    b = Individual("1", np.array([0.2, 0.6]), None)
    c = Individual("2", np.array([0.38, 0.5]), None)
    d = Individual("3", np.array([0.6, 0.25]), None)
    f = Individual("4", np.array([0.4, 0.6]), None)

    pop = [b, c, d, f]

    survival = _RDominanceSurvival(directions=['min', 'min'], population_size=4,
                                   random_state=get_random_state(),
                                   ref_point=reference_point, weights=[0.5, 0.5], threshold=0.3)

    def cmp(x1, x2, directions=None):
        return survival.dominate(x1, x2, pop=pop)

    assert not cmp(b, c)
    assert cmp(b, d)
    assert cmp(c, d)

    assert cmp(b, f)
    assert cmp(c, f)

    assert not cmp(d, f)

    # nondominated_set = calc_nondominated_set(, dominate_func=cmp)
    #
    # assert len(nondominated_set) == 2
    # assert b in nondominated_set
    # assert c in nondominated_set



# if __name__ == '__main__':
#     Test_RNGGA2().reproce_nsga2_training()
