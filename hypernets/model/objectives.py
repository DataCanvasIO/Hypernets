import abc

from hypernets.core.objective import Objective


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

    def call(self, trial, estimator, X_test, y_test, **kwargs):
        from hypernets.tabular.metrics import calc_score
        # TODO: convert probabilities to prediction
        y_pred = estimator.predict(X_test)
        y_proba = estimator.predict_proba(X_test)
        scores = calc_score(y_true=y_test, y_preds=y_pred, y_proba=y_proba, metrics=[self.name])
        return scores.get(self.name)


class FeatureComplexityObjective(ComplexityObjective):

    def call(self, trial, estimator, y_test, **kwargs):
        pass
