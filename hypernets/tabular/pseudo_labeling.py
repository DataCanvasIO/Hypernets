# -*- coding:utf-8 -*-
"""

"""
from collections import Counter

import numpy as np

from hypernets.tabular import dask_ex as dex
from hypernets.utils import logging

logger = logging.get_logger(__name__)

_STRATEGY_THRESHOLD = 'threshold'
_STRATEGY_QUANTILE = 'quantile'
_STRATEGY_NUMBER = 'number'
_STRATEGY_DEFAULT = _STRATEGY_THRESHOLD

_DEFAULT_THRESHOLD = 0.8
_DEFAULT_QUANTILE = 0.8
_DEFAULT_TOP_PERCENT = 0.2


def detect_strategy(strategy, threshold=None, quantile=None, number=None):
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


def sample_by_pseudo_labeling(X_test, classes, proba, strategy,
                              threshold=None, quantile=None, number=None):
    assert len(classes) == proba.shape[-1] > 1

    strategy, threshold, quantile, number = \
        detect_strategy(strategy, threshold=threshold, quantile=quantile, number=number)
    if strategy == _STRATEGY_NUMBER and isinstance(number, float) and 0 < number < 1:
        proba_shape = dex.compute(proba.shape)[0] if dex.is_dask_object(proba) else proba.shape
        number = int(proba_shape[0] / proba_shape[1] * _DEFAULT_TOP_PERCENT)
        if number < 10:
            number = 10

    if dex.is_dask_dataframe(X_test):
        fn = _sample_by_dask
    else:
        fn = _sample_by_sk

    r = fn(X_test, classes, proba, strategy, threshold, quantile, number)
    if logger.is_info_enabled():
        if dex.is_dask_object(r[1]):
            y = dex.compute(r[1])[0]
        else:
            y = r[1]
        logger.info(f'extract pseudo labeling samples (strategy={strategy}): {Counter(y)}')

    return r


def _sample_by_sk(X_test, classes, proba, strategy, threshold, quantile, number):
    mx = proba.max(axis=1, keepdims=True)
    proba = np.where(proba < mx, 0, proba)

    if strategy is None or strategy == _STRATEGY_THRESHOLD:
        selected = (proba >= threshold)
    elif strategy == _STRATEGY_NUMBER:
        pos = proba.shape[0] - number
        i = np.argsort(np.argsort(proba, axis=0), axis=0)
        selected = np.logical_and(i >= pos, proba > 0)
    elif strategy == _STRATEGY_QUANTILE:
        qs = np.nanquantile(np.where(proba > 0, proba, np.nan), quantile, axis=0)
        selected = (proba >= qs)
    else:
        raise ValueError(f'Unsupported strategy: {strategy}')

    pred = (selected * np.arange(1, len(classes) + 1)).max(axis=1) - 1
    idx = np.argwhere(pred >= 0).ravel()

    X_pseudo = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
    y_pseudo = np.take(np.array(classes), pred[idx], axis=0)

    return X_pseudo, y_pseudo


def _sample_by_dask(X_test, classes, proba, strategy, threshold, quantile, number):
    da = dex.da
    mx = proba.max(axis=1, keepdims=True)
    proba = da.where(proba < mx, 0, proba)
    proba = dex.make_chunk_size_known(proba)

    if strategy is None or strategy == _STRATEGY_THRESHOLD:
        selected = (proba >= threshold)
    elif strategy == _STRATEGY_NUMBER:
        if proba.numblocks[0] > 1:
            proba = proba.rechunk(proba.shape)
        selected = proba.map_blocks(__select_top, number, meta=np.array((), np.bool))
    elif strategy == _STRATEGY_QUANTILE:
        qs = []
        for i in range(proba.shape[-1]):
            c = proba[:, i]
            ci = da.argwhere(c > 0)
            ci = dex.make_chunk_size_known(ci).ravel()
            t = c[ci]
            qs.append(da.percentile(t, quantile * 100))
        qs = dex.compute(qs)[0]
        selected = (proba >= np.array(qs).ravel())
    else:
        raise ValueError(f'Unsupported strategy: {strategy}')

    pred = (selected * np.arange(1, len(classes) + 1)).max(axis=1) - 1
    idx = da.argwhere(pred >= 0)
    idx = dex.make_chunk_size_known(idx).ravel()

    if dex.is_dask_dataframe(X_test):
        X_test_values = X_test.to_dask_array(lengths=True)
        X_test_values = X_test_values[idx]
        X_pseudo = dex.dd.from_dask_array(X_test_values, columns=X_test.columns,
                                          meta=X_test._meta)
    else:
        X_pseudo = X_test[idx]
    y_pseudo = da.take(np.array(classes), pred[idx], axis=0)

    return X_pseudo, y_pseudo


def __select_top(chunk, number):
    pos = chunk.shape[0] - number
    i = np.argsort(np.argsort(chunk, axis=0), axis=0)
    result = np.logical_and(i >= pos, chunk > 0)
    return result
