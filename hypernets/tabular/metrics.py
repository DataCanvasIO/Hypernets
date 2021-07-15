# -*- coding:utf-8 -*-
"""

"""

import numpy as np
from dask import dataframe as dd
from sklearn import metrics as sk_metrics

from hypernets.utils import logging, const
from . import dask_ex as dex

logger = logging.getLogger(__name__)

_DASK_METRICS = ('accuracy', 'logloss')


def calc_score(y_true, y_preds, y_proba=None, metrics=('accuracy',), task=const.TASK_BINARY, pos_label=1,
               classes=None, average=None):
    data_list = (y_true, y_proba, y_preds)
    if any(map(dex.is_dask_object, data_list)):
        if all(map(dex.is_dask_object, data_list)) and len(set(metrics).difference(set(_DASK_METRICS))) == 0:
            fn = _calc_score_dask
        else:
            y_true, y_proba, y_preds = \
                [y.compute() if dex.is_dask_object(y) else y for y in data_list]
            fn = _calc_score_sklean
    else:
        fn = _calc_score_sklean

    return fn(y_true, y_preds, y_proba, metrics, task=task, pos_label=pos_label,
              classes=classes, average=average)


def task_to_average(task):
    if task == const.TASK_MULTICLASS:
        average = 'macro'
    else:
        average = 'binary'
    return average


def _calc_score_sklean(y_true, y_preds, y_proba=None, metrics=('accuracy',), task=const.TASK_BINARY, pos_label=1,
                       classes=None, average=None):
    score = {}
    if y_proba is None:
        y_proba = y_preds
    if len(y_proba.shape) == 2 and y_proba.shape[-1] == 1:
        y_proba = y_proba.reshape(-1)
    if len(y_preds.shape) == 2 and y_preds.shape[-1] == 1:
        y_preds = y_preds.reshape(-1)

    if average is None:
        average = task_to_average(task)

    recall_options = dict(average=average, labels=classes)
    if pos_label is not None:
        recall_options['pos_label'] = pos_label

    for metric in metrics:
        if callable(metric):
            score[metric.__name__] = metric(y_true, y_preds)
        else:
            metric_lower = metric.lower()
            if metric_lower == 'auc':
                if len(y_proba.shape) == 2:
                    if task == const.TASK_MULTICLASS:
                        score[metric] = sk_metrics.roc_auc_score(y_true, y_proba, multi_class='ovo', labels=classes)
                    else:
                        score[metric] = sk_metrics.roc_auc_score(y_true, y_proba[:, 1])
                else:
                    score[metric] = sk_metrics.roc_auc_score(y_true, y_proba)
            elif metric_lower == 'accuracy':
                if y_preds is None:
                    score[metric] = 0
                else:
                    score[metric] = sk_metrics.accuracy_score(y_true, y_preds)
            elif metric_lower == 'recall':
                score[metric] = sk_metrics.recall_score(y_true, y_preds, **recall_options)
            elif metric_lower == 'precision':
                score[metric] = sk_metrics.precision_score(y_true, y_preds, **recall_options)
            elif metric_lower == 'f1':
                score[metric] = sk_metrics.f1_score(y_true, y_preds, **recall_options)
            elif metric_lower == 'mse':
                score[metric] = sk_metrics.mean_squared_error(y_true, y_preds)
            elif metric_lower == 'mae':
                score[metric] = sk_metrics.mean_absolute_error(y_true, y_preds)
            elif metric_lower == 'msle':
                score[metric] = sk_metrics.mean_squared_log_error(y_true, y_preds)
            elif metric_lower in {'rmse', 'rootmeansquarederror', 'root_mean_squared_error'}:
                score[metric] = np.sqrt(sk_metrics.mean_squared_error(y_true, y_preds))
            elif metric_lower == 'r2':
                score[metric] = sk_metrics.r2_score(y_true, y_preds)
            elif metric_lower in {'logloss', 'log_loss'}:
                score[metric] = sk_metrics.log_loss(y_true, y_proba, labels=classes)

    return score


def _calc_score_dask(y_true, y_preds, y_proba=None, metrics=('accuracy',), task=const.TASK_BINARY, pos_label=1,
                     classes=None, average=None):
    import dask_ml.metrics as dm_metrics

    def to_array(name, value):
        if value is None:
            return value

        if isinstance(value, (dd.DataFrame, dd.Series)):
            value = value.values

        if len(value.shape) == 2 and value.shape[-1] == 1:
            value = value.reshape(-1)

        value = dex.make_chunk_size_known(value)
        return value

    score = {}

    y_true = to_array('y_true', y_true)
    y_preds = to_array('y_preds', y_preds)
    y_proba = to_array('y_proba', y_proba)

    if y_true.chunks[0] != y_preds.chunks[0]:
        logger.debug(f'rechunk y_preds with {y_true.chunks[0]}')
        y_preds = y_preds.rechunk(chunks=y_true.chunks[0])

    if y_proba is None:
        y_proba = y_preds
    elif y_true.chunks[0] != y_proba.chunks[0]:
        if len(y_proba.chunks) > 1:
            chunks = (y_true.chunks[0],) + y_proba.chunks[1:]
        else:
            chunks = y_true.chunks
        logger.debug(f'rechunk y_proba with {chunks}')
        y_proba = y_proba.rechunk(chunks=chunks)

    for metric in metrics:
        if callable(metric):
            score[metric.__name__] = metric(y_true, y_preds)
        else:
            metric_lower = metric.lower()
            if metric_lower == 'accuracy':
                score[metric] = dm_metrics.accuracy_score(y_true, y_preds)
            elif metric_lower == 'logloss':
                ll = dm_metrics.log_loss(y_true, y_proba, labels=classes)
                if hasattr(ll, 'compute'):
                    ll = ll.compute()
                score[metric] = ll
            else:
                logger.warning(f'unknown metric: {metric}')
    return score


def metric_to_scoring(metric, task=const.TASK_BINARY, pos_label=None):
    assert isinstance(metric, str)

    metric2scoring = {
        'auc': 'roc_auc_ovo',
        'accuracy': 'accuracy',
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1',
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'msle': 'neg_mean_squared_log_error',
        'rmse': 'neg_root_mean_squared_error',
        'rootmeansquarederror': 'neg_root_mean_squared_error',
        'root_mean_squared_error': 'neg_root_mean_squared_error',
        'r2': 'r2',
        'logloss': 'neg_log_loss',
    }
    metric2fn = {
        'recall': sk_metrics.recall_score,
        'precision': sk_metrics.precision_score,
        'f1': sk_metrics.f1_score,
    }
    metric_lower = metric.lower()
    if metric_lower not in metric2scoring.keys() and metric_lower not in metric2fn.keys():
        raise ValueError(f'Not found matching scoring for {metric}')

    if metric_lower in metric2fn.keys():
        scoring = sk_metrics.make_scorer(metric2fn[metric_lower], pos_label=pos_label, average=task_to_average(task))
    else:
        scoring = sk_metrics.get_scorer(metric2scoring[metric_lower])

    return scoring
