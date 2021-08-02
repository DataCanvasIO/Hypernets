# -*- coding:utf-8 -*-
"""

"""
import math
import os
import pickle
import sys

import numpy as np
import psutil
from dask import dataframe as dd
from joblib import Parallel, delayed
from sklearn import metrics as sk_metrics

from hypernets.utils import const, infer_task_type, logging
from . import dask_ex as dex

logger = logging.get_logger(__name__)

_is_os_windows = sys.platform.find('win') >= 0

_MIN_BATCH_SIZE = 100000

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
        options = dict(average=task_to_average(task))
        if pos_label is not None:
            options['pos_label'] = pos_label
        scoring = sk_metrics.make_scorer(metric2fn[metric_lower], **options)
    else:
        scoring = sk_metrics.get_scorer(metric2scoring[metric_lower])

    return scoring


def evaluate(estimator, X, y, metrics, *, task=None, pos_label=None, classes=None,
             average=None, threshold=0.5, n_jobs=-1):
    assert classes is None or isinstance(classes, (list, tuple, np.ndarray))
    if isinstance(estimator, str):
        assert os.path.exists(estimator), f'Not found {estimator}'

    if task is None and classes is not None and len(classes) >= 2:
        task = const.TASK_BINARY if len(classes) == 2 else const.TASK_MULTICLASS
    if task is None:
        task, c2 = _detect_task(estimator, y)
        if classes is None or len(classes) < 2:
            classes = c2

    n_jobs = _detect_jobs(X, n_jobs)
    if task in {const.TASK_BINARY, const.TASK_MULTICLASS}:
        proba = predict_proba(estimator, X, n_jobs=n_jobs)
        pred = proba2predict(proba, task=task, threshold=threshold, classes=classes)
    else:
        pred = predict(estimator, X, n_jobs=n_jobs)
        proba = None

    if task == const.TASK_BINARY and pos_label is None:
        pos_label = classes[-1]

    if logger.is_info_enabled():
        logger.info(f'calc_score {metrics}, task={task}, pos_label={pos_label}, classes={classes}, average={average}')
    scores = calc_score(y_true=y, y_preds=pred, y_proba=proba, metrics=metrics,
                        classes=classes, pos_label=pos_label, average=average)

    return scores


def predict_proba(estimator, X, *, n_jobs=-1):
    if isinstance(estimator, str):
        assert os.path.exists(estimator), f'Not found {estimator}'

    n_jobs = _detect_jobs(X, n_jobs)
    if logger.is_info_enabled():
        logger.info(f'predict_proba with n_jobs={n_jobs}')
    proba = _call_predict(estimator, 'predict_proba', X, n_jobs=n_jobs)

    return proba


def predict(estimator, X, *, task=None, classes=None, threshold=0.5, n_jobs=-1):
    assert classes is None or isinstance(classes, (list, tuple, np.ndarray))
    if isinstance(estimator, str):
        assert os.path.exists(estimator), f'Not found {estimator}'

    if task is None and classes is not None and len(classes) >= 2:
        task = const.TASK_BINARY if len(classes) == 2 else const.TASK_MULTICLASS
    if task is None:
        task, c2 = _detect_task(estimator, None)
        if classes is None or len(classes) < 2:
            classes = c2

    if task == const.TASK_REGRESSION:
        if logger.is_info_enabled():
            logger.info(f'predict with n_jobs={n_jobs}')
        pred = _call_predict(estimator, 'predict', X, n_jobs=n_jobs)
    else:
        proba = predict_proba(estimator, X, n_jobs=n_jobs)
        if task is None and (len(proba.shape) < 2 or proba.shape[1] == 1):
            task = const.TASK_REGRESSION
        pred = proba2predict(proba, task=task, threshold=threshold, classes=classes)
    return pred


def proba2predict(proba, *, task=None, threshold=0.5, classes=None):
    assert len(proba.shape) <= 2

    def is_one_dim(x):
        return len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1)

    if logger.is_info_enabled():
        logger.info(f'proba2predict with task={task}, classes={classes}, threshold={threshold}')

    if len(proba.shape) == 0:  # empty
        return proba

    if task == const.TASK_BINARY and is_one_dim(proba):
        proba = dex.fix_binary_predict_proba_result(proba)

    if task == const.TASK_REGRESSION or is_one_dim(proba):  # regression
        return proba

    if proba.shape[-1] > 2:  # multiclass
        predict = proba.argmax(axis=-1)
    else:  # binary
        predict = (proba[:, -1] > threshold).astype(np.int32)

    if classes is not None:
        if dex.is_dask_object(predict):
            predict = dex.da.take(np.array(classes), predict, axis=0)
        else:
            predict = np.take(np.array(classes), predict, axis=0)

    return predict


def _detect_jobs(X, n_jobs):
    if callable(getattr(X, 'compute', None)):  # dask data frame
        return 1

    assert X.shape[0] > 0, f'Not found data.'

    if n_jobs <= 0:
        n_jobs = math.ceil(X.shape[0] / _MIN_BATCH_SIZE)

    cores = psutil.cpu_count()
    if n_jobs > cores:
        n_jobs = cores

    return n_jobs


def _detect_task(estimator, y):
    if isinstance(estimator, str):
        with open(estimator, 'rb') as f:
            estimator = pickle.load(f)

    task = None
    classes = None

    if task is None and hasattr(estimator, 'task'):
        task = getattr(estimator, 'task', None)

    if task is None and type(estimator).__name__.find('Pipeline') >= 0 and hasattr(estimator, 'steps'):
        task = getattr(estimator.steps[-1][1], 'task', None)

    if hasattr(estimator, 'classes_'):
        classes = getattr(estimator, 'classes_', None)
        if not (isinstance(classes, (list, tuple, np.ndarray)) and len(classes) > 1):
            classes = None
        if task is None and classes is not None:
            task = const.TASK_BINARY if len(classes) == 2 else const.TASK_MULTICLASS

    if classes is None and type(estimator).__name__.find('Pipeline') >= 0 and hasattr(estimator, 'steps'):
        classes = getattr(estimator.steps[-1][1], 'classes_', None)
        if not (isinstance(classes, (list, tuple, np.ndarray)) and len(classes) > 1):
            classes = None
        if task is None and classes is not None:
            task = const.TASK_BINARY if len(classes) == 2 else const.TASK_MULTICLASS

    if task is None and y is not None:
        task, c2 = infer_task_type(y)
        if classes is None:
            classes = c2

    return task, classes


def _load_and_run(estimator, fn_name, df):
    if isinstance(estimator, str):
        logger.info(f'load estimator {estimator}')
        with open(estimator, 'rb') as f:
            estimator = pickle.load(f)

    fn = getattr(estimator, fn_name)
    assert callable(fn)

    logger.info(f'call {fn_name}')
    result = fn(df)

    return result


def _call_predict(estimator, fn_name, df, n_jobs=1):
    if n_jobs > 1:
        batch_size = math.ceil(df.shape[0] / n_jobs)
        df_parts = [df[i:i + batch_size].copy() for i in range(df.index.start, df.index.stop, batch_size)]
        options = dict(backend='multiprocessing') if _is_os_windows else dict(prefer='processes')
        pss = Parallel(n_jobs=n_jobs, **options)(delayed(_load_and_run)(estimator, fn_name, x) for x in df_parts)

        if len(pss[0].shape) > 1:
            result = np.vstack(pss)
        else:
            result = np.hstack(pss)
    else:
        result = _load_and_run(estimator, fn_name, df)

    return result
