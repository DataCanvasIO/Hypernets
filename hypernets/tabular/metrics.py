# -*- coding:utf-8 -*-
"""

"""

import numpy as np
from dask import dataframe as dd
from sklearn import metrics as sk_metrics

from hypernets.utils import logging
from . import dask_ex as dex

logger = logging.getLogger(__name__)

_DASK_METRICS = ('accuracy', 'logloss')


def calc_score(y_true, y_preds, y_proba=None, metrics=('accuracy',), task='binary', pos_label=1):
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

    return fn(y_true, y_preds, y_proba, metrics, task, pos_label)


def _calc_score_sklean(y_true, y_preds, y_proba=None, metrics=('accuracy',), task='binary', pos_label=1):
    score = {}
    if y_proba is None:
        y_proba = y_preds
    if len(y_proba.shape) == 2 and y_proba.shape[-1] == 1:
        y_proba = y_proba.reshape(-1)
    if len(y_preds.shape) == 2 and y_preds.shape[-1] == 1:
        y_preds = y_preds.reshape(-1)
    for metric in metrics:
        if callable(metric):
            score[metric.__name__] = metric(y_true, y_preds)
        else:
            metric = metric.lower()
            if task == 'multiclass':
                average = 'micro'
            else:
                average = 'binary'

            if metric == 'auc':
                if len(y_proba.shape) == 2:
                    if task == 'multiclass':
                        score['auc'] = sk_metrics.roc_auc_score(y_true, y_proba, multi_class='ovo')
                    else:
                        score['auc'] = sk_metrics.roc_auc_score(y_true, y_proba[:, 1])
                else:
                    score['auc'] = sk_metrics.roc_auc_score(y_true, y_proba)

            elif metric == 'accuracy':
                if y_proba is None:
                    score['accuracy'] = 0
                else:
                    score['accuracy'] = sk_metrics.accuracy_score(y_true, y_preds)
            elif metric == 'recall':
                score['recall'] = sk_metrics.recall_score(y_true, y_preds, average=average, pos_label=pos_label)
            elif metric == 'precision':
                score['precision'] = sk_metrics.precision_score(y_true, y_preds, average=average, pos_label=pos_label)
            elif metric == 'f1':
                score['f1'] = sk_metrics.f1_score(y_true, y_preds, average=average, pos_label=pos_label)

            elif metric == 'mse':
                score['mse'] = sk_metrics.mean_squared_error(y_true, y_preds)
            elif metric == 'mae':
                score['mae'] = sk_metrics.mean_absolute_error(y_true, y_preds)
            elif metric == 'msle':
                score['msle'] = sk_metrics.mean_squared_log_error(y_true, y_preds)
            elif metric == 'rmse':
                score['rmse'] = np.sqrt(sk_metrics.mean_squared_error(y_true, y_preds))
            elif metric == 'rootmeansquarederror':
                score['rootmeansquarederror'] = np.sqrt(sk_metrics.mean_squared_error(y_true, y_preds))
            elif metric == 'r2':
                score['r2'] = sk_metrics.r2_score(y_true, y_preds)
            elif metric == 'logloss':
                score['logloss'] = sk_metrics.log_loss(y_true, y_proba)

    return score


def _calc_score_dask(y_true, y_preds, y_proba=None, metrics=('accuracy',), task='binary', pos_label=1):
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
            metric = metric.lower()
            if metric == 'accuracy':
                score[metric] = dm_metrics.accuracy_score(y_true, y_preds)
            elif metric == 'logloss':
                ll = dm_metrics.log_loss(y_true, y_proba)
                if hasattr(ll, 'compute'):
                    ll = ll.compute()
                score[metric] = ll
            else:
                logger.warning(f'unknown metric: {metric}')
    return score


def metric_to_scoring(metric):
    mapping = {
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
        'r2': 'r2',
        'logloss': 'neg_log_loss',
    }
    if metric not in mapping.keys():
        raise ValueError(f'Not found matching scoring for {metric}')

    return mapping[metric]
