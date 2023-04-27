import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import pytest

from hypernets.model.objectives import NumOfFeatures, PredictionPerformanceObjective, PredictionObjective, calc_psi


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

        return lr, X_test, y_test

    def create_cv_models(self):
        X_train, X_test, y_train, y_test = self.create_mock_dataset()

        lr1 = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
        lr2 = DecisionTreeRegressor(max_depth=2).fit(X_train, y_train)
        lr3 = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)

        return [lr1, lr2, lr3], [X_test] * 3, [y_test] * 3


class TestNumOfFeatures(BaseTestWithinModel):

    def test_call(self):
        lr, X_test, y_test = self.create_model()
        nof = NumOfFeatures()
        score = nof.call(trial=None, estimator=lr, X_test=X_test, y_test=y_test)
        assert score < 1  # only 2 features used
        features = nof.get_used_features(estimator=lr, X_test=X_test)
        assert 'log' in set(features) or 'exp' in set(features)

    def test_call_cross_validation(self):
        estimators, X_tests, y_tests = self.create_cv_models()
        nof = NumOfFeatures()
        score = nof.call_cross_validation(trial=None, estimators=estimators, X_tests=X_tests, y_tests=y_tests)
        assert 0 < score < 1  # only 2 features used
        features = nof.get_cv_used_features(estimators=estimators, X_tests=X_tests)
        assert 'log' in set(features) or 'exp' in set(features)


class TestPredictionPerformanceObjective(BaseTestWithinModel):

    def test_call(self):
        lr, X_test, y_test = self.create_model()
        ppo = PredictionPerformanceObjective()
        score = ppo.call(trial=None,  estimator=lr, X_test=X_test, y_test=y_test)
        assert score is not None

    def test_call_cross_validation(self):
        estimators, X_tests, y_tests = self.create_cv_models()
        ppo = PredictionPerformanceObjective()
        score = ppo.call_cross_validation(trial=None, estimators=estimators, X_tests=X_tests, y_tests=y_tests)
        assert score is not None


class FakeEstimator:
    def __init__(self, class_, proba):
        self.classes_ = class_
        self.proba = proba

    def predict_proba(self, X, **kwargs):
        return self.proba

    def predict(self, X, **kwargs):
        return self.proba[:, 1] > 0.5


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

        objective = PredictionObjective.create(name=metric_name, force_minimize=force_minimize)
        score = objective.call_cross_validation(trial=None, estimators=estimators,
                                                X_tests=X_tests, y_tests=y_trues)
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


class TestPSIObjective:

    def test_calc_psi(self):
        x_array = np.random.random((100, 1))
        y_array = np.random.random((100, 1))

        psi1 = calc_psi(x_array, y_array, n_bins=10, eps=1e-6)
        psi2 = calc_psi(x_array * 10, y_array * 5, n_bins=10, eps=1e-6)
        assert psi1 > 0
        assert psi2 > 0
        assert psi2 > psi1
        print(psi1)

