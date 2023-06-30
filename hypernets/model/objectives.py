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


def calc_psi(x_array, y_array, n_bins=10, eps=1e-6):
    def calc_ratio(y_proba):
        y_proba_1d = y_proba.reshape(1, -1)
        ratios = []
        for i, interval in enumerate(intervals):
            if i == len(interval) - 1:
                # include the probability==1
                n_samples = (y_proba_1d[np.where((y_proba_1d >= interval[0]) & (y_proba_1d <= interval[1]))]).shape[0]
            else:
                n_samples = (y_proba_1d[np.where((y_proba_1d >= interval[0]) & (y_proba_1d < interval[1]))]).shape[0]
            ratio = n_samples / y_proba.shape[0]
            if ratio == 0:
                ratios.append(eps)
            else:
                ratios.append(ratio)
        return np.array(ratios)

    assert x_array.ndim == 2 and y_array.ndim == 2, "please reshape to 2-d ndarray"

    # stats max and min
    all_data = np.vstack((x_array, y_array))
    max_val = np.max(all_data)
    min_val = np.min(all_data)

    distance = (max_val - min_val) / n_bins
    intervals = [(i * distance + min_val, (i+1) * distance + min_val) for i in range(n_bins)]
    train_ratio = calc_ratio(x_array)
    test_ratio = calc_ratio(y_array)
    return np.sum((train_ratio - test_ratio) * np.log(train_ratio / test_ratio))



def detect_used_features(estimator, X_data, sample_size=1000):

    if sample_size >= X_data.shape[0]:
        sample_size = X_data.shape[0]
    else:
        sample_size = sample_size

    D: pd.DataFrame = X_data.sample(sample_size, random_state=random_state)
    # D.reset_index(inplace=True, drop=True)

    y_pred = estimator.predict(D.copy())  # predict can modify D
    NF = []
    for feature in X_data.columns:
        unique = X_data[feature].unique()
        n_unique = len(unique)
        if n_unique < 2: # skip constant feature
            continue
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


class PSIObjective(Objective):

    def __init__(self, n_bins=10, task=const.TASK_BINARY, average='macro', eps=1e-6):
        super(PSIObjective, self).__init__('psi', 'min', need_train_data=True, need_val_data=False, need_test_data=True)
        if task == const.TASK_MULTICLASS and average != 'macro':
            raise RuntimeError("only 'macro' average is supported currently")
        if task not in [const.TASK_BINARY, const.TASK_MULTICLASS, const.TASK_REGRESSION]:
            raise RuntimeError(f"unseen task type {task}")
        self.n_bins = n_bins
        self.task = task
        self.average = average
        self.eps = eps

    def _evaluate(self, trial, estimator, X_train, y_train, X_val, y_val, X_test=None, **kwargs) -> float:
        return self._get_psi_score(estimator, X_train, X_test)

    def _get_psi_score(self, estimator, X_train, X_test):
        def to_2d(array_data):
            if array_data.ndim == 1:
                return array_data.reshape((-1, 1))
            else:
                return array_data
        if self.task == const.TASK_BINARY:
            train_proba = estimator.predict_proba(X_train)
            test_proba = estimator.predict_proba(X_test)
            return float(calc_psi(to_2d(train_proba[:, 1]), to_2d(test_proba[:, 1])))
        elif self.task == const.TASK_REGRESSION:
            train_result = to_2d(estimator.predict(X_train))
            test_result = to_2d(estimator.predict(X_test))
            return float(calc_psi(train_result, test_result))
        elif self.task == const.TASK_MULTICLASS:
            train_proba = estimator.predict_proba(X_train)
            test_proba = estimator.predict_proba(X_test)
            psis = [float(calc_psi(to_2d(train_proba[:, i]), to_2d(test_proba[:, 1]))) for i in
                    range(train_proba.shape[1])]
            return float(np.mean(psis))
        else:
            raise RuntimeError(f"unseen task type {self.task}")

    def _evaluate_cv(self, trial, estimator, X_trains, y_trains, X_vals, y_vals, X_test=None, **kwargs) -> float:
        X_train = pd.concat(X_trains, axis=0)
        return self._get_psi_score(estimator, X_train=X_train, X_test=X_test)


class ElapsedObjective(Objective):

    def __init__(self):
        super(ElapsedObjective, self).__init__(name='elapsed', direction='min', need_train_data=False,
                                               need_val_data=False, need_test_data=False)

    def _evaluate(self, trial, estimator, X_train, y_train, X_val, y_val, X_test=None, **kwargs) -> float:
        return trial.elapsed

    def _evaluate_cv(self, trial, estimators, X_trains, y_trains, X_vals, y_vals, X_test=None, **kwargs) -> float:
        return trial.elapsed


class PredictionPerformanceObjective(Objective):

    def __init__(self):
        super(PredictionPerformanceObjective, self).__init__('pred_perf', 'min', need_train_data=False,
                                                             need_val_data=True,
                                                             need_test_data=False)

    def _evaluate(self, trial, estimator, X_train, y_train, X_val, y_val, X_test=None, **kwargs) -> float:
        t1 = time.time()
        estimator.predict(X_val)
        return time.time() - t1

    def _evaluate_cv(self, trial, estimator, X_trains, y_trains, X_vals, y_vals, X_test=None, **kwargs) -> float:
        t1 = time.time()
        estimator.predict(pd.concat(X_vals, axis=0))
        return time.time() - t1


class CVWrapperEstimator:

    def __init__(self, estimators, x_vals, y_vals):
        self.estimators = estimators
        self.x_vals = x_vals
        self.y_vals = y_vals

    @property
    def classes_(self):
        return self.estimators[0].classes_

    @property
    def _estimator_type(self):
        try:
            if len(self.classes_) > 1:
                return 'classifier'
            else:
                return 'regressor'
        except:
            return 'regressor'

    def predict(self, X, **kwargs):
        rows = 0
        for x_val in self.x_vals:
            assert x_val.ndim == 2
            assert X.shape[1] == x_val.shape[1]
            rows = x_val.shape[0] + rows
        assert rows == X.shape[0]

        proba = []
        for estimator, x_val in zip(self.estimators, self.x_vals):
            proba.extend(estimator.predict(x_val))
        return np.asarray(proba)

    def predict_proba(self, X, **kwargs):
        rows = 0
        for x_val in self.x_vals:
            assert x_val.ndim == 2
            assert X.shape[1] == x_val.shape[1]
            rows = x_val.shape[0] + rows
        assert rows == X.shape[0]

        proba = []
        for estimator, x_val in zip(self.estimators, self.x_vals):
            proba.extend(estimator.predict_proba(x_val))
        return np.asarray(proba)


class PredictionObjective(Objective):

    def __init__(self, name, scorer, direction=None):
        if direction is None:
            direction = 'max' if scorer._sign > 0 else 'min'

        super(PredictionObjective, self).__init__(name, direction=direction, need_train_data=False,
                                                  need_val_data=True, need_test_data=False)
        self._scorer = scorer

    @staticmethod
    def _default_score_args(force_minimize):
        # for positive metrics which are the bigger, the better
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

    def _evaluate(self, trial, estimator, X_train, y_train, X_val, y_val, X_test=None, **kwargs) -> float:
        value = self._scorer(estimator, X_val, y_val)
        return value

    def _evaluate_cv(self, trial, estimator, X_trains, y_trains, X_vals, y_vals, X_test=None, **kwargs) -> float:

        estimator = CVWrapperEstimator(estimator.cv_models_, X_vals, y_vals)
        X_test = pd.concat(X_vals, axis=0)

        y_test = np.vstack(y_test.values.reshape((-1, 1)) if isinstance(y_test, pd.Series) else y_test.reshape((-1, 1)) for y_test in y_vals).reshape(-1, )
        return self._scorer(estimator, X_test, y_test)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, scorer={self._scorer}, direction={self.direction})"


class NumOfFeatures(Objective):
    """Detect the number of features used (NF)

    References:
        [1] Molnar, Christoph, Giuseppe Casalicchio, and Bernd Bischl. "Quantifying model complexity via functional decomposition for better post-hoc interpretability." Machine Learning and Knowledge Discovery in Databases: International Workshops of ECML PKDD 2019, Würzburg, Germany, September 16–20, 2019, Proceedings, Part I. Springer International Publishing, 2020.
    """

    def __init__(self, sample_size=1000):
        super(NumOfFeatures, self).__init__('nf', 'min')
        self.sample_size = sample_size

    def _evaluate(self, trial, estimator, X_train, y_train, X_val, y_val, X_test=None, **kwargs) -> float:
        features = self.get_used_features(estimator=estimator, X_data=X_val)
        return len(features) / len(X_val.columns)

    def _evaluate_cv(self, trial, estimator, X_trains, y_trains, X_vals, y_vals, X_test=None, **kwargs) -> float:
        used_features = self.get_cv_used_features(estimator, X_vals)
        return len(used_features) / len(X_vals[0].columns)

    def get_cv_used_features(self, estimator, X_datas):
        used_features = []
        for X_data in X_datas:
            features = self.get_used_features(estimator, X_data)
            used_features.extend(features)
        return list(set(used_features))

    def get_used_features(self, estimator, X_data):
        return detect_used_features(estimator, X_data, self.sample_size)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, sample_size={self.sample_size}, direction={self.direction})"


def create_objective(name, **kwargs):
    def copy_opt(opt_names):
        for opt_name in opt_names:
            if opt_name in kwargs:
                opts[opt_name] = kwargs.get(opt_name)

    name = name.lower()
    opts = {}

    if name == 'elapsed':
        return ElapsedObjective()
    elif name == 'nf':
        copy_opt(['sample_size'])
        return NumOfFeatures(**opts)
    elif name == 'psi':
        copy_opt(['n_bins', 'task', 'average', 'eps'])
        return PSIObjective(**opts)
    elif name == 'pred_perf':
        return PredictionPerformanceObjective()
    else:
        copy_opt(['task', 'pos_label', 'force_minimize'])
        return PredictionObjective.create(name, **kwargs)
