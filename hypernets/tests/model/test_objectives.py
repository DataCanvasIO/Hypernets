import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import pytest

from hypernets.core import set_random_state, get_random_state
from hypernets.examples.plain_model import PlainSearchSpace, PlainModel
from hypernets.model.objectives import NumOfFeatures, PredictionPerformanceObjective, PredictionObjective, calc_psi, \
    PSIObjective, create_objective
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from hypernets.searchers import NSGAIISearcher
from hypernets.searchers.genetic import create_recombination
from hypernets.tests.searchers.test_nsga2_searcher import get_bankdata
from hypernets.utils import const


class BaseTestWithinModel:

    def create_mock_dataset(self):
        X = np.random.random((10000, 4))
        df = pd.DataFrame(data=X, columns= [str("c_%s" % i) for i in range(4)])
        y = np.random.random(10000)
        df['exp'] = np.exp(y)
        df['log'] = np.log(y)
        return train_test_split(df, y, test_size=0.5)

    def create_model(self):
        X_train, X_test, y_train, y_test = self.create_mock_dataset()

        lr = DecisionTreeRegressor(max_depth=2)
        lr.fit(X_train, y_train)

        return lr, X_train, X_test, y_train, y_test

    def create_cv_models(self):
        X_train, X_test, y_train, y_test = self.create_mock_dataset()

        lr1 = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
        lr2 = DecisionTreeRegressor(max_depth=2).fit(X_train, y_train)
        lr3 = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)

        return [lr1, lr2, lr3], [X_train] * 3, [y_train] * 3, [X_test] * 3, [y_test] * 3


class TestNumOfFeatures(BaseTestWithinModel):

    def test_call(self):
        lr, X_train, X_test, y_train, y_test = self.create_model()
        nof = NumOfFeatures()
        score = nof.evaluate(trial=None, estimator=lr, X_val=X_test, y_val=y_test, X_train=None, y_train=None, X_test=None)
        assert score < 1  # only 2 features used
        features = nof.get_used_features(estimator=lr, X_data=X_test)
        assert 'log' in set(features) or 'exp' in set(features)

    def test_call_cross_validation(self):
        estimators, X_trians, y_trains, X_tests, y_tests = self.create_cv_models()
        nof = NumOfFeatures()
        score = nof.evaluate_cv(trial=None, estimator=estimators[0], X_trains=X_trians, y_trains=y_trains,
                    X_vals=X_tests, y_vals=y_tests, X_test=None)
        assert 0 < score < 1  # only 2 features used
        features = nof.get_cv_used_features(estimator=estimators[0], X_datas=X_tests)
        assert 'log' in set(features) or 'exp' in set(features)


class FakeCVEstimator:

    def __init__(self, estimators):
        self.cv_models_ = estimators

    def predict(self, *args, **kwargs):
        return self.cv_models_[0].predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return self.cv_models_[0].predict_proba(*args, **kwargs)

    @property
    def _estimator_type(self):
        return 'classifier'

class TestPredictionPerformanceObjective(BaseTestWithinModel):

    def test_call(self):
        lr, X_train, X_test, y_train, y_test = self.create_model()
        ppo = PredictionPerformanceObjective()
        score = ppo.evaluate(trial=None, estimator=lr, X_val=X_test, y_val=y_test, X_train=None, y_train=None, X_test=None)
        assert score is not None

    def test_call_cross_validation(self):
        estimators, X_trians, y_trains, X_tests, y_tests = self.create_cv_models()
        ppo = PredictionPerformanceObjective()
        FakeCVEstimator(estimators)
        score = ppo.evaluate_cv(trial=None, estimator=FakeCVEstimator(estimators),
                                X_trains=None, y_trains=None, X_vals=X_tests, y_vals=y_tests, X_test=None)
        assert score is not None


class FakeEstimator:
    def __init__(self, class_, proba):
        self.classes_ = class_
        self.proba = proba

    def predict_proba(self, X, **kwargs):
        return self.proba

    def predict(self, X, **kwargs):
        return self.proba[:, 1] > 0.5

    @property
    def _estimator_type(self):
        return 'classifier'


class TestPredictionObjective:

    def create_objective(self, metric_name, force_minimize):
        y_true = np.array([1, 1, 0, 1]).reshape((4, 1))
        y_proba = np.array([[0.2, 0.8], [0.1, 0.9], [0.9, 0.1], [0.3, 0.7]]).reshape((4, 2))
        estimator = FakeEstimator(class_=np.array([0, 1]), proba=y_proba)
        objective = PredictionObjective.create(name=metric_name, force_minimize=force_minimize)
        score = objective.get_score()(estimator=estimator, X=None, y_true=y_true)
        return objective, score

    def create_cv_objective(self, metric_name, force_minimize):
        n_rows = 6
        y_trues = [np.array([1, 1, 0, 1, 0, 1]).reshape((n_rows, 1))] * 3
        y_proba1 = np.array([[0.2, 0.8], [0.1, 0.9], [0.9, 0.1], [0.3, 0.7], [0.9, 0.1], [0.3, 0.7]]).reshape(
            (n_rows, 2))
        y_proba2 = np.array([[0.3, 0.7], [0.2, 0.8], [0.8, 0.2], [0.4, 0.6], [0.8, 0.2], [0.4, 0.6]]).reshape(
            (n_rows, 2))
        y_proba3 = np.array([[0.4, 0.6], [0.3, 0.8], [0.7, 0.3], [0.5, 0.5], [0.7, 0.3], [0.5, 0.5]]).reshape(
            (n_rows, 2))

        estimator1 = FakeEstimator(class_=np.array([0, 1]), proba=y_proba1)
        estimator2 = FakeEstimator(class_=np.array([0, 1]), proba=y_proba2)
        estimator3 = FakeEstimator(class_=np.array([0, 1]), proba=y_proba3)
        estimators = [estimator1, estimator2, estimator3]
        X_tests = [pd.DataFrame(data=np.random.random((6, 2)), columns=['c1', 'c2'])] * 3
        y_tests_array = np.random.binomial(n=1, p=0.5, size=(3, n_rows))
        y_tests = []
        for _ in y_tests_array:
            y_tests.append(_)

        objective = PredictionObjective.create(name=metric_name, force_minimize=force_minimize)
        score = objective.evaluate_cv(trial=None, estimator=FakeCVEstimator(estimators),
                                      X_trains=None, y_trains=None, X_vals=X_tests, y_vals=y_tests, X_test=None)
        return objective, score

    @pytest.mark.parametrize('metric_name', ['logloss', 'auc', 'f1', 'precision', 'recall', 'accuracy'])
    @pytest.mark.parametrize('force_minimize', [True, False])
    @pytest.mark.parametrize('cv', [True, False])
    def test_create(self, metric_name: str, force_minimize: bool, cv: bool):
        if cv:
            objective, score = self.create_cv_objective(metric_name, force_minimize)
        else:
            objective, score = self.create_objective(metric_name, force_minimize)
        assert objective.name == metric_name

        if force_minimize:
            assert objective.direction == "min"
        else:
            if metric_name == "logloss":
                assert objective.direction == "min"
            else:
                assert objective.direction == "max"

        if force_minimize:
            if metric_name == "logloss":
                assert score > 0
            else:
                assert score < 0
        else:
            assert score > 0


class TestPSIObjective(BaseTestWithinModel):

    def test_calc_psi(self):
        x_array = np.random.random((100, 1))
        y_array = np.random.random((100, 1))
        psi1 = calc_psi(x_array, y_array, n_bins=10, eps=1e-6)
        psi2 = calc_psi(x_array * 10, y_array * 5, n_bins=10, eps=1e-6)
        assert psi1 > 0
        assert psi2 > 0
        assert psi2 > psi1
        print(psi1)

    def test_call(self):
        lr, X_train, X_test, y_train, y_test = self.create_model()
        po = PSIObjective(n_bins=10, task=const.TASK_REGRESSION, average='macro', eps=1e-6)
        score = po.evaluate(trial=None, estimator=lr, X_val=None, y_val=None, X_train=X_train,
                            y_train=y_train, X_test=X_test)
        assert score is not None

    def test_call_cross_validation(self):
        estimators, X_trians, y_trains, X_tests, y_tests = self.create_cv_models()
        ppo = PSIObjective(n_bins=10, task=const.TASK_REGRESSION, average='macro', eps=1e-6)
        score = ppo.evaluate_cv(trial=None, estimator=estimators[0], X_trains=X_trians,
                                y_trains=y_trains, X_vals=None, y_vals=None, X_test=X_tests[0])
        assert score is not None

    def test_search(self):
        set_random_state(1234)
        X_train, y_train, X_test, y_test = get_bankdata()
        recombination_ins = create_recombination('shuffle', random_state=get_random_state())
        search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)
        rs = NSGAIISearcher(search_space, objectives=[PredictionObjective.create('accuracy'),
                                                      create_objective('psi')],
                            recombination=recombination_ins, population_size=3)

        # the given reward_metric is in order to ensure SOO working, make it's the same as metrics in MOO searcher
        hk = PlainModel(rs, task='binary', transformer=MultiLabelEncoder, reward_metric='logloss')

        hk.search(X_train, y_train, X_test, y_test, X_test=X_test.copy(), max_trials=5, cv=True)

        len(hk.history.trials)
        assert hk.get_best_trial()

    def test_search_multi_classification(self):
        set_random_state(1234)

        df = dsutils.load_glass_uci()
        df.columns = [f'x_{c}' for c in df.columns]
        y = df.pop('x_10')

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

        recombination_ins = create_recombination('shuffle', random_state=get_random_state())
        search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)
        rs = NSGAIISearcher(search_space, objectives=[PredictionObjective.create('accuracy'),
                                                      create_objective('psi')],
                            recombination=recombination_ins, population_size=3)

        # the given reward_metric is in order to ensure SOO working, make it's the same as metrics in MOO searcher
        hk = PlainModel(rs, task='binary', transformer=MultiLabelEncoder, reward_metric='logloss')

        hk.search(X_train, y_train, X_test, y_test, X_test=X_test.copy(), max_trials=5, cv=True)

        len(hk.history.trials)
        assert hk.get_best_trial()

