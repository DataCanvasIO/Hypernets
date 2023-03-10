import abc

from hypernets.core.objective import Objective
from hypernets.utils import const


class ComplexityObjective(Objective, metaclass=abc.ABCMeta):
    pass


class PerformanceObjective(Objective, metaclass=abc.ABCMeta):
    pass


class ElapsedObjective(PerformanceObjective):

    def __init__(self):
        super(ElapsedObjective, self).__init__(name='elapsed', direction='min')

    def call(self, trial, estimator, y_test, **kwargs):
        return trial.elapsed


class PredictionObjective(PerformanceObjective):

    def __init__(self, name, scorer):
        direction = 'max' if scorer._sign > 0 else 'min'
        super(PredictionObjective, self).__init__(name, direction=direction)
        self._scorer = scorer

    def call(self, trial, estimator, X_test, y_test, **kwargs):
        import numpy as np
        #y_pred = estimator.predict(X_test)
        #y_proba = estimator.predict_proba(X_test)
        return np.absolute(self._scorer(estimator, X_test, y_test))

    @staticmethod
    def create(name, task=const.TASK_BINARY, pos_label=None):
        from hypernets.tabular.metrics import metric_to_scoring
        scorer = metric_to_scoring(metric=name, task=task, pos_label=pos_label)
        return PredictionObjective(name, scorer)


class FeatureComplexityObjective(ComplexityObjective):

    def call(self, trial, estimator, y_test, **kwargs):
        pass


def create_objective(name, **kwargs):
    # objective factory, exclude PredictionObjective now
    if name == 'elapsed':
        return ElapsedObjective()
    else:
        # check it in sklearn metrics
        # po = PredictionObjective(name, **kwargs)
        # return po
        raise RuntimeError(f"unseen objective name {name}")
