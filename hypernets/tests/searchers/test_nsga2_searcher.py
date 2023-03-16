import pytest

from hypernets.model.objectives import ElapsedObjective, PredictionObjective
from hypernets.searchers.nsga_searcher import NSGAIISearcher, NSGAIndividual, RankAndCrowdSortSurvival, \
    RDominanceSurvival

from sklearn.preprocessing import LabelEncoder
from hypernets.core.random_state import set_random_state, get_random_state
from hypernets.examples.plain_model import PlainSearchSpace, PlainModel
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from sklearn.model_selection import train_test_split

from hypernets.searchers.moo import pareto_dominate, calc_nondominated_set
from hypernets.searchers.genetic import Individual

set_random_state(1234)

from hypernets.core.callbacks import *


def test_fast_non_dominated_sort():
    survival = RankAndCrowdSortSurvival(directions=['min', 'min'], random_state=get_random_state())
    i1 = NSGAIndividual("1", np.array([0.1, 0.3]), None)
    i2 = NSGAIndividual("2", np.array([0.2, 0.3]), None)

    l = survival.fast_non_dominated_sort([i1, i2])
    assert len(l) == 2

    assert l[0][0] == i1
    assert l[1][0] == i2

    # first rank has two element
    i3 = NSGAIndividual("3", np.array([0.3, 0.1]), None)
    l = survival.fast_non_dominated_sort([i1, i2, i3])
    assert len(l) == 2
    assert i1 in l[0]
    assert i3 in l[0]
    assert l[1][0] == i2

    i4 = NSGAIndividual("4", np.array([0.25, 0.3]), None)
    l = survival.fast_non_dominated_sort([i1, i2, i3, i4])
    assert len(l) == 3
    assert l[2][0] == i4


def test_crowd_distance_sort():
    survival = RankAndCrowdSortSurvival(directions=['min', 'min'], random_state=get_random_state())
    i1 = NSGAIndividual("1", np.array([0.10, 0.30]), None)
    i2 = NSGAIndividual("2", np.array([0.11, 0.25]), None)
    i3 = NSGAIndividual("3", np.array([0.12, 0.19]), None)
    i4 = NSGAIndividual("4", np.array([0.13, 0.10]), None)

    pop = survival.crowding_distance_assignment([i1, i2, i3, i4])  # i1, i2, i3, i4 are in the same rank

    assert i1.distance == i4.distance == float("inf")  # i1 & i4 are always selected
    assert i3.distance > i2.distance  # i3 is more sparsity


@pytest.mark.parametrize('recombination', ["shuffle", "uniform", "single_point"])
def test_nsga2_training(recombination: str):
    df = dsutils.load_bank().sample(1000)
    df['y'] = LabelEncoder().fit_transform(df['y'])

    df.drop(['id'], axis=1, inplace=True)
    X_train, X_test = train_test_split(df, test_size=0.8, random_state=1234)

    y_train = X_train.pop('y')
    y_test = X_test.pop('y')

    search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)
    rs = NSGAIISearcher(search_space, objectives=[ElapsedObjective(),
                                                  PredictionObjective.create('logloss')],
                        recombination=recombination, population_size=3)

    hk = PlainModel(rs, task='binary', callbacks=[SummaryCallback()], transformer=MultiLabelEncoder)

    hk.search(X_train, y_train, X_test, y_test, max_trials=5)

    len(hk.history.trials)
    assert hk.get_best_trial()


def test_non_consistent_direction():

    df = dsutils.load_bank().sample(1000)
    df['y'] = LabelEncoder().fit_transform(df['y'])

    df.drop(['id'], axis=1, inplace=True)
    X_train, X_test = train_test_split(df, test_size=0.8, random_state=1234)

    y_train = X_train.pop('y')
    y_test = X_test.pop('y')

    search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)
    rs = NSGAIISearcher(search_space, objectives=[ElapsedObjective(),
                                                  PredictionObjective.create('auc')],
                        recombination='single_point', population_size=10)

    hk = PlainModel(rs, task='binary', callbacks=[SummaryCallback()], transformer=MultiLabelEncoder)

    hk.search(X_train, y_train, X_test, y_test, max_trials=30)

    len(hk.history.trials)
    assert hk.get_best_trial()

    ns = rs.get_nondominated_set()
    assert ns


#
# class TestRDominanceSurvival:
#     @classmethod
#     def setup_class(cls):
#         # data setting from fig. 3 in the paper
#         reference_point = [0.2, 0.4]
#         weights = np.array([0.5, 0.5])  # ignore effect of weights
#
#         scores = np.array([[0.1, 0.9], [0.2, 0.6], [0.38, 0.5], [0.6, 0.25],
#                           [0.3, 0.95], [0.4, 0.6], [0.7, 0.45], [0.8, 0.4],
#                           [0.9, 0.3], [0.5, 0.9], [0.6, 0.75], [0.95, 0.6],
#                           [0.8, 0.9], [0.9, 0.8], [0.9, 0.9], [0.95, 0.9]])
#
#         pop = [NSGAIndividual(str(i), score, None) for i, score in enumerate(scores)]
#         cls.pop = pop
#
#         cmp = partial(RDominanceSurvival.r_dominance, ref_point=reference_point, weights=weights,
#                       directions=['min', 'min'], pop=scores, threshold=0.3)
#
#         cls._dominate = cmp
#
#         cls.survival = RDominanceSurvival(directions=['min', 'min'], random_state=get_random_state(),
#                                           ref_point=reference_point,
#                                           weights=weights, dominance_threshold=0.3)
#
#         i_b = NSGAIndividual("1", pop[0], None)
#         i_c = NSGAIndividual("2", pop[1], None)
#         i_d = NSGAIndividual("3", pop[2], None)
#         i_f = NSGAIndividual("4", pop[3], None)
#         cls.individuals = [i_b, i_c, i_d, i_f]
#
#     def test_dominate(self):
#         a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p = self.pop
#         # _dominate = partial(self.survival.dominate, pop=self.pop)
#
#         def _dominate(x1, x2):
#             return self.survival.dominate(x1=x1.scores, x2=x2.scores, pop=self.pop)
#
#         assert not _dominate(b, c)
#         assert _dominate(b, d)
#         assert _dominate(c, d)
#
#         assert _dominate(b, f)
#         assert _dominate(c, f)
#
#         assert not _dominate(d, f)
#         assert not _dominate(f, d)
#
#     def test_nondominate_set(self):
#         i_b = Individual("1", self.pop[0], None)
#         i_c = Individual("2", self.pop[1], None)
#         i_d = Individual("3", self.pop[2], None)
#         i_f = Individual("4", self.pop[3], None)
#
#         nondominated_set = calc_nondominated_set([i_b, i_c, i_d, i_f], dominate_func=self._dominate)
#
#         assert len(nondominated_set) == 2
#         assert i_b in nondominated_set
#         assert i_c in nondominated_set
#
#     def test_rank(self):
#         F = self.survival.fast_non_dominated_sort(self.individuals, directions=['min', 'min'])
#         assert len(F) == 2  # two levels
#
#         assert self.individuals[0] in F[0]
#         assert self.individuals[1] in F[0]
#
#         assert self.individuals[3] in F[1]
#         assert self.individuals[4] in F[1]
#


class TestRDominanceSurvival:
    @classmethod
    def setup_class(cls):
        # data setting from fig. 3 in the paper
        reference_point = [0.05, 0.05]
        weights = np.array([0.5, 0.5])  # ignore effect of weights

        scores = np.array([[0.1, 0.1], [0.1, 0.15], [0.2, 0.2], [0.3, 0.3]])

        pop = [NSGAIndividual(str(i), score, None) for i, score in enumerate(scores)]
        cls.pop = pop

        cls.survival = RDominanceSurvival(directions=['min', 'min'], random_state=get_random_state(),
                                          ref_point=reference_point,
                                          weights=weights, dominance_threshold=0.3)

    def test_dominate(self):
        a, b, c, d = self.pop
        # _dominate = partial(self.survival.dominate, pop=self.pop)

        def _dominate(x1, x2):
            return self.survival.dominate(x1=x1.scores, x2=x2.scores, pop=self.pop)

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

    b = np.array([0.2, 0.6])
    c = np.array([0.38, 0.5])
    d = np.array([0.6, 0.25])
    f = np.array([0.4, 0.6])

    def cmp(x1, x2, directions=None):
        return RDominanceSurvival.r_dominance(x1, x2, ref_point=reference_point, weights=np.array([0.5, 0.5]),
                                               directions=['min', 'min'], pop=np.array([b, c, d, f]), threshold=0.3)

    assert not cmp(b, c)
    assert cmp(b, d)
    assert cmp(c, d)

    assert cmp(b, f)
    assert cmp(c, f)

    assert not cmp(d, f)

    i_b = Individual("1", np.array([0.2, 0.6]), None)
    i_c = Individual("2", np.array([0.38, 0.5]), None)
    i_d = Individual("3", np.array([0.6, 0.25]), None)
    i_f = Individual("4", np.array([0.4, 0.6]), None)

    nondominated_set = calc_nondominated_set([i_b, i_c, i_d, i_f], dominate_func=cmp)

    assert len(nondominated_set) == 2
    assert i_b in nondominated_set
    assert i_c in nondominated_set

