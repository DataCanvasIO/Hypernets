import abc
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, make_scorer, roc_auc_score, accuracy_score, \
    f1_score, precision_score, recall_score

from hypernets.core import get_random_state
from hypernets.core.objective import Objective
from hypernets.utils import const
from hypernets.tabular.metrics import metric_to_scoring

random_state = get_random_state()


class ComplexityObjective(Objective, metaclass=abc.ABCMeta):
    pass


class PerformanceObjective(Objective, metaclass=abc.ABCMeta):
    pass


class ElapsedObjective(PerformanceObjective):

    def __init__(self):
        super(ElapsedObjective, self).__init__(name='elapsed', direction='min')

    def call(self, trial, estimator, y_test, **kwargs):
        return trial.elapsed


class PredictionPerformanceObjective(Objective):
    def __init__(self):
        super(PredictionPerformanceObjective, self).__init__('pred_perf', 'min')

    def call(self, trial, estimator, X_test, y_test, **kwargs) -> float:
        t1 = time.time()
        estimator.predict(X_test)
        return time.time() - t1


class PredictionObjective(PerformanceObjective):

    def __init__(self, name, scorer, direction=None):
        if direction is None:
            direction = 'max' if scorer._sign > 0 else 'min'

        super(PredictionObjective, self).__init__(name, direction=direction)
        self._scorer = scorer

    def call(self, trial, estimator, X_test, y_test, **kwargs):
        value = self._scorer(estimator, X_test, y_test)
        return value

    @staticmethod
    def _default_score_args(force_minimize):
        # for positive metrics which are the greater the better
        if force_minimize:
            greater_is_better = False
            direction = 'min'
        else:
            greater_is_better = True
            direction = 'max'
        return greater_is_better, direction

    @staticmethod
    def create_auc(name, force_minimize):

        greater_is_better, direction = PredictionObjective._default_score_args(force_minimize)

        scorer = make_scorer(roc_auc_score, greater_is_better=greater_is_better,
                             needs_threshold=True)  # average=average

        return PredictionObjective(name, scorer, direction=direction)

    @staticmethod
    def create_f1(name, force_minimize, pos_label, average):
        greater_is_better, direction = PredictionObjective._default_score_args(force_minimize)

        scorer = make_scorer(f1_score, greater_is_better=greater_is_better, needs_threshold=False,
                             pos_label=pos_label, average=average)
        return PredictionObjective(name, scorer, direction=direction)

    @staticmethod
    def create_precision(name, force_minimize, pos_label, average):
        greater_is_better, direction = PredictionObjective._default_score_args(force_minimize)

        scorer = make_scorer(precision_score, greater_is_better=greater_is_better, needs_threshold=False,
                             pos_label=pos_label, average=average)
        return PredictionObjective(name, scorer, direction=direction)

    @staticmethod
    def create_recall(name, force_minimize, pos_label, average):
        greater_is_better, direction = PredictionObjective._default_score_args(force_minimize)
        scorer = make_scorer(recall_score, greater_is_better=greater_is_better, needs_threshold=False,
                             pos_label=pos_label, average=average)
        return PredictionObjective(name, scorer, direction=direction)

    @staticmethod
    def create_accuracy(name, force_minimize):

        greater_is_better, direction = PredictionObjective._default_score_args(force_minimize)

        scorer = make_scorer(accuracy_score, greater_is_better=greater_is_better, needs_threshold=False)

        return PredictionObjective(name, scorer, direction=direction)

    @staticmethod
    def create(name, task=const.TASK_BINARY, pos_label=1, force_minimize=False):
        default_average = 'macro' if task == const.TASK_MULTICLASS else 'binary'

        lower_name = name.lower()
        if lower_name == 'logloss':
            # Note: the logloss score in sklearn is negative of naive logloss to maximize optimization
            scorer = make_scorer(log_loss, greater_is_better=True, needs_proba=True)  # let _sign > 0
            return PredictionObjective(name, scorer, direction='min')
        elif lower_name == 'auc':
            return PredictionObjective.create_auc(lower_name, force_minimize)

        elif lower_name == 'f1':
            return PredictionObjective.create_f1(lower_name, force_minimize,
                                                 pos_label=pos_label, average=default_average)

        elif lower_name == 'precision':
            return PredictionObjective.create_precision(lower_name, force_minimize,
                                                        pos_label=pos_label, average=default_average)

        elif lower_name == 'recall':
            return PredictionObjective.create_recall(lower_name, force_minimize,
                                                     pos_label=pos_label, average=default_average)
        elif lower_name == 'accuracy':
            return PredictionObjective.create_accuracy(lower_name, force_minimize)
        else:
            scorer = metric_to_scoring(metric=name, task=task, pos_label=pos_label)
            return PredictionObjective(name, scorer)

    def get_score(self):
        return self._scorer

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, scorer={self._scorer}, direction={self.direction})"


class FeatureComplexityObjective(ComplexityObjective):

    def call(self, trial, estimator, y_test, **kwargs):
        pass


class NumOfFeatures(ComplexityObjective):
    """Detect the number of features used (NF)

    References:
        [1] Molnar, Christoph, Giuseppe Casalicchio, and Bernd Bischl. "Quantifying model complexity via functional decomposition for better post-hoc interpretability." Machine Learning and Knowledge Discovery in Databases: International Workshops of ECML PKDD 2019, Würzburg, Germany, September 16–20, 2019, Proceedings, Part I. Springer International Publishing, 2020.
    """

    def __init__(self, sample_size=1000):
        super(NumOfFeatures, self).__init__('nf', 'min')
        self.sample_size = sample_size

    def call(self, trial, estimator, X_test, y_test, **kwargs) -> float:
        features = self.get_used_features(estimator=estimator, X_test=X_test)
        return len(features) / len(X_test.columns)

    def get_used_features(self, estimator, X_test):
        if self.sample_size >= X_test.shape[0]:
            sample_size = X_test.shape[0]
        else:
            sample_size = self.sample_size

        D: pd.DataFrame = X_test.sample(sample_size, random_state=random_state)
        # D.reset_index(inplace=True, drop=True)

        y_pred = estimator.predict(D.copy())  # predict can modify D
        NF = []
        for feature in X_test.columns:
            unique = X_test[feature].unique()
            n_unique = len(unique)
            samples_inx = random_state.randint(low=0, high=n_unique - 1, size=D.shape[0])
            # transform inx that does not contain self
            mapped_inx = []

            for i, value in zip(samples_inx, D[feature].values):
                j = int(np.where(unique == value)[0][0])
                if i >= j:
                    mapped_inx.append(i + 1)
                else:
                    mapped_inx.append(i)

            D_ = D.copy()
            D_[feature] = unique[mapped_inx]

            if (D_[feature] == D[feature]).values.any():
                raise RuntimeError("some samples have not been replaced by different value")

            y_pred_modified = estimator.predict(D_)
            if (y_pred != y_pred_modified).any():
                NF.append(feature)
            del D_

        return NF

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, sample_size={self.sample_size}, direction={self.direction})"


def create_objective(name, force_minimize=False, sample_size=2000, task=const.TASK_BINARY, pos_label=1, **kwargs):
    name = name.lower()
    if name == 'elapsed':
        return ElapsedObjective()
    elif name == 'nf':
        return NumOfFeatures(sample_size=sample_size)
    elif name == 'pred_perf':
        return PredictionPerformanceObjective()
    else:
        return PredictionObjective.create(name, force_minimize=force_minimize, task=task, pos_label=pos_label, **kwargs)
