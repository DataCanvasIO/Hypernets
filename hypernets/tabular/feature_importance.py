# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import numpy as np
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.utils import Bunch

from hypernets.tabular import dask_ex as dex
from hypernets.utils import logging
from .cfg import TabularCfg as c

logger = logging.get_logger(__name__)

_STRATEGY_THRESHOLD = 'threshold'
_STRATEGY_QUANTILE = 'quantile'
_STRATEGY_NUMBER = 'number'
_STRATEGY_DEFAULT = _STRATEGY_THRESHOLD

_DEFAULT_THRESHOLD = 0.1
_DEFAULT_QUANTILE = 0.2
_DEFAULT_TOP_PERCENT = 0.8


def detect_strategy(strategy=None, threshold=None, quantile=None, number=None):
    if strategy is None:
        if threshold is not None:
            strategy = _STRATEGY_THRESHOLD
        elif number is not None:
            strategy = _STRATEGY_NUMBER
        elif quantile is not None:
            strategy = _STRATEGY_QUANTILE
        else:
            strategy = _STRATEGY_DEFAULT

    if strategy == _STRATEGY_THRESHOLD:
        if threshold is None:
            threshold = _DEFAULT_THRESHOLD
        assert 0 < threshold < 1.0
    elif strategy == _STRATEGY_NUMBER:
        if number is None:
            number = _DEFAULT_TOP_PERCENT
    elif strategy == _STRATEGY_QUANTILE:
        if quantile is None:
            quantile = _DEFAULT_QUANTILE
        assert 0 < quantile < 1.0
    else:
        raise ValueError(f'Unsupported strategy: {strategy}')

    return strategy, threshold, quantile, number


def select_by_feature_importance(feature_importance, strategy=None,
                                 threshold=None, quantile=None, number=None):
    assert isinstance(feature_importance, (list, tuple, np.ndarray)) and len(feature_importance) > 0

    strategy, threshold, quantile, number = detect_strategy(strategy, threshold, quantile, number)
    if strategy == _STRATEGY_NUMBER and isinstance(number, float) and 0 < number < 1.0:
        number = len(feature_importance) * _DEFAULT_TOP_PERCENT

    feature_importance = np.array(feature_importance)
    idx = np.arange(len(feature_importance))

    if strategy == _STRATEGY_THRESHOLD:
        selected = np.where(np.where(feature_importance >= threshold, idx, -1) >= 0)[0]
    elif strategy == _STRATEGY_QUANTILE:
        q = np.quantile(feature_importance, quantile)
        selected = np.where(np.where(feature_importance >= q, idx, -1) >= 0)[0]
    elif strategy == _STRATEGY_NUMBER:
        pos = len(feature_importance) - number
        sorted = np.argsort(np.argsort(feature_importance))
        selected = np.where(sorted >= pos)[0]
    else:
        raise ValueError(f'Unsupported strategy: {strategy}')

    unselected = list(set(range(len(feature_importance))) - set(selected))
    unselected = np.array(unselected)

    return selected, unselected


def permutation_importance_batch(estimators, X, y, scoring=None, n_repeats=5,
                                 n_jobs=None, random_state=None):
    """Evaluate the importance of features of a set of estimators

    Parameters
    ----------
    estimator : list
        A set of estimators that has already been :term:`fitted` and is compatible
        with :term:`scorer`.

    X : ndarray or DataFrame, shape (n_samples, n_features)
        Data on which permutation importance will be computed.

    y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
        Targets for supervised or `None` for unsupervised.

    scoring : string, callable or None, default=None
        Scorer to use. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer is used.

    n_repeats : int, default=5
        Number of times to permute a feature.

    n_jobs : int or None, default=None
        The number of jobs to use for the computation.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, or None, default=None
        Pseudo-random number generator to control the permutations of each
        feature. See :term:`random_state`.

    Returns
    -------
    result : Bunch
        Dictionary-like object, with attributes:

        importances_mean : ndarray, shape (n_features, )
            Mean of feature importance over `n_repeats`.
        importances_std : ndarray, shape (n_features, )
            Standard deviation over `n_repeats`.
        importances : ndarray, shape (n_features, n_repeats)
            Raw permutation importance scores.
    """
    importances = []

    if dex.is_dask_dataframe(X):
        X_shape = dex.compute(X.shape)[0]
        permutation_importance = dex.permutation_importance
    else:
        X_shape = X.shape
        permutation_importance = sk_permutation_importance

    if X_shape[0] > c.permutation_importance_sample_limit:
        if logger.is_info_enabled():
            logger.info(f'{X_shape[0]} rows data found, sample to {c.permutation_importance_sample_limit}')
        frac = c.permutation_importance_sample_limit / X_shape[0]
        X, _, y, _ = dex.train_test_split(X, y, train_size=frac, random_state=random_state)

    if n_jobs is None:
        n_jobs = c.joblib_njobs

    for i, est in enumerate(estimators):
        if logger.is_info_enabled():
            logger.info(f'score permutation importance by estimator {i}/{len(estimators)}')
        importance = permutation_importance(est, X.copy(), y.copy(),
                                            scoring=scoring, n_repeats=n_repeats, n_jobs=n_jobs,
                                            random_state=random_state)
        importances.append(importance.importances)

    importances = np.reshape(np.stack(importances, axis=2), (X.shape[1], -1), 'F')
    bunch = Bunch(importances_mean=np.mean(importances, axis=1),
                  importances_std=np.std(importances, axis=1),
                  importances=importances,
                  columns=X.columns.to_list())
    return bunch
