import abc

import numpy as np
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

    def __init__(self, name, scorer, direction=None):
        if direction is None:
            direction = 'max' if scorer._sign > 0 else 'min'

        super(PredictionObjective, self).__init__(name, direction=direction)
        self._scorer = scorer

    def call(self, trial, estimator, X_test, y_test, **kwargs):
        #y_pred = estimator.predict(X_test)
        #y_proba = estimator.predict_proba(X_test)
        value = self._scorer(estimator, X_test, y_test)
        return value

    @staticmethod
    def create(name, task=const.TASK_BINARY, pos_label=None):
        from sklearn.metrics import log_loss, make_scorer, roc_auc_score, accuracy_score

        if name.lower() == 'logloss':
            # Note: the logloss score in sklearn is negative of naive logloss to maximize optimization
            scorer = make_scorer(log_loss, greater_is_better=True, needs_proba=True)  # let _sign>0
            return PredictionObjective(name, scorer, direction='min')
        elif name.lower() == 'roc_auc':
            scorer = make_scorer(roc_auc_score, greater_is_better=False, needs_threshold=True)
            return PredictionObjective(name, scorer, direction='min')
        elif name.lower() == 'accuracy':
            scorer = make_scorer(accuracy_score, greater_is_better=False, needs_threshold=False)
            return PredictionObjective(name, scorer, direction='min')
        else:
            from hypernets.tabular.metrics import metric_to_scoring
            scorer = metric_to_scoring(metric=name, task=task, pos_label=pos_label)
            return PredictionObjective(name, scorer, direction='min')



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
