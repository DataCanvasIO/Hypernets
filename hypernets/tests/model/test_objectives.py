import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from hypernets.model.objectives import NumOfFeatures, PredictionPerformanceObjective, PredictionObjective
from sklearn.metrics import log_loss
import numpy as np
from sklearn.metrics import get_scorer
import pytest


class TestNumOfFeatures:

    def create_mock_dataset(self):
        X = np.random.random((10000, 4))
        df = pd.DataFrame(data=X, columns= [str("c_%s" % i) for i in range(4)])
        y = np.random.random(10000)
        df['exp'] = np.exp(y)
        df['log'] = np.log(y)
        return train_test_split(df, y, test_size=0.5)

    def test_call(self):
        X_train, X_test, y_train, y_test = self.create_mock_dataset()

        lr = DecisionTreeRegressor(max_depth=2)
        lr.fit(X_train, y_train)

        nof = NumOfFeatures()
        score = nof.call(trial=None,  estimator=lr, X_test=X_test, y_test=y_test)

        assert score < 1  # only 2 features used

        features = nof.get_used_features(estimator=lr, X_test=X_test)
        assert 'log' in set(features) or 'exp' in set(features)


class FakeEstimator:
    def __init__(self, class_, proba):
        self.classes_ = class_
        self.proba = proba

    def predict_proba(self, X, **kwargs):
        return self.proba

    def predict(self, X, **kwargs):
        return self.proba[:, 1] > 0.5


class TestPredictionObjective:

    @classmethod
    def setup_class(cls):
        cls.y_true = np.array([1, 1, 0, 1]).reshape((4, 1))
        y_proba = np.array([[0.2, 0.8], [0.1, 0.9], [0.9, 0.1], [0.3, 0.7]]).reshape((4, 2))
        cls.estimator = FakeEstimator(class_=np.array([0, 1]), proba=y_proba)

    @pytest.mark.parametrize('metric_name', ['logloss', 'auc', 'f1', 'precision', 'recall', 'accuracy'])
    @pytest.mark.parametrize('force_minimize', [True, False])
    def test_create(self, metric_name: str, force_minimize: bool):

        objective = PredictionObjective.create(name=metric_name, force_minimize=force_minimize)
        score = objective.get_score()(estimator=self.estimator, X=None, y_true=self.y_true)

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
