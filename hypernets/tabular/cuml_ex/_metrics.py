# -*- coding:utf-8 -*-
"""

"""
import cudf
import cupy as cp
import numpy as np
from cuml import metrics as cu_metrics
from cuml.common.array import CumlArray
from hypernets.utils import const, logging
from ..metrics import Metrics

logger = logging.get_logger(__name__)

_CUML_METRICS = {'accuracy', 'logloss', 'log_loss',
                 'mse', 'mae', 'msle',
                 'rmse', 'rootmeansquarederror', 'root_mean_squared_error',
                 'r2'}


def _to_dtype(y, dtype):
    if y is not None:
        if hasattr(y, 'dtype') and not y.dtype == dtype:
            if isinstance(y, CumlArray):
                y = cp.asarray(y, dtype=dtype)
            else:
                y = y.astype(dtype)
        elif hasattr(y, 'dtypes') and not all(y.dtypes == dtype):
            y = y.astype(dtype)

    return y


def _calc_score_cuml(y_true, y_preds, y_proba=None, metrics=('accuracy',), task=const.TASK_BINARY, pos_label=1,
                     classes=None, average=None):
    if y_proba is None:
        y_proba = y_preds
    if len(y_proba.shape) == 2 and y_proba.shape[-1] == 1:
        y_proba = y_proba.reshape(-1)
    if len(y_preds.shape) == 2 and y_preds.shape[-1] == 1:
        y_preds = y_preds.reshape(-1)

    y_true = _to_dtype(y_true, 'float64')
    y_preds = _to_dtype(y_preds, 'float64')
    y_proba = _to_dtype(y_proba, 'float64')

    if task == const.TASK_REGRESSION:
        if isinstance(y_true, cudf.Series):
            y_true = y_true.values
        if isinstance(y_preds, cudf.Series):
            y_preds = y_preds.values
        if isinstance(y_proba, cudf.Series):
            y_proba = y_proba.values

    scores = {}
    for metric in metrics:
        if callable(metric):
            scores[metric.__name__] = metric(y_true, y_preds)
        else:
            metric_lower = metric.lower()
            if metric_lower == 'auc':
                if len(y_proba.shape) == 2:
                    # if task == const.TASK_MULTICLASS:
                    #     s = cu_metrics.roc_auc_score(y_true, y_proba, multi_class='ovo', labels=classes)
                    # else:
                    #     s = cu_metrics.roc_auc_score(y_true, y_proba[:, 1])
                    s = cu_metrics.roc_auc_score(y_true, y_proba[:, 1])
                else:
                    s = cu_metrics.roc_auc_score(y_true, y_proba)
            elif metric_lower == 'accuracy':
                if y_preds is None:
                    s = 0
                else:
                    s = cu_metrics.accuracy_score(y_true, y_preds)
            # elif metric_lower == 'recall':
            #     s = cu_metrics.recall_score(y_true, y_preds, **recall_options)
            # elif metric_lower == 'precision':
            #     s = cu_metrics.precision_score(y_true, y_preds, **recall_options)
            # elif metric_lower == 'f1':
            #     s = cu_metrics.f1_score(y_true, y_preds, **recall_options)
            elif metric_lower == 'mse':
                s = cu_metrics.mean_squared_error(y_true, y_preds)
            elif metric_lower == 'mae':
                s = cu_metrics.mean_absolute_error(y_true, y_preds)
            elif metric_lower == 'msle':
                s = cu_metrics.mean_squared_log_error(y_true, y_preds)
            elif metric_lower in {'rmse', 'rootmeansquarederror', 'root_mean_squared_error'}:
                s = cu_metrics.mean_squared_error(y_true, y_preds, squared=False)
            elif metric_lower == 'r2':
                s = cu_metrics.r2_score(y_true, y_preds)
            elif metric_lower in {'logloss', 'log_loss'}:
                # s = cu_metrics.log_loss(y_true, y_proba, labels=classes)
                s = cu_metrics.log_loss(y_true, y_proba)
            else:
                logger.warning(f'unknown metric: {metric}')
                continue
            if isinstance(s, cp.ndarray):
                s = float(cp.asnumpy(s))
            scores[metric] = s
    return scores


class CumlMetrics(Metrics):
    @staticmethod
    def calc_score(y_true, y_preds, y_proba=None, metrics=('accuracy',), task=const.TASK_BINARY, pos_label=1,
                   classes=None, average=None):
        from ._toolbox import CumlToolBox

        data_list = (y_true, y_proba, y_preds)
        if any(map(CumlToolBox.is_cuml_object, data_list)):
            diff_metrics = set(metrics).difference(_CUML_METRICS)
            if y_true.dtype.kind in {'i', 'f'} and \
                    all(map(lambda _: _ is None or CumlToolBox.is_cuml_object(_), data_list)) and \
                    len(diff_metrics) == 0:
                # failed to get auc in Rapids v21.10
                # (len(diff_metrics) == 0 or (diff_metrics == {'auc'} and task == const.TASK_BINARY)):
                fn = _calc_score_cuml
            else:
                y_true, y_proba, y_preds = CumlToolBox.to_local(*data_list)
                fn = Metrics.calc_score
        else:
            fn = Metrics.calc_score

        return fn(y_true, y_preds, y_proba, metrics, task=task, pos_label=pos_label,
                  classes=classes, average=average)
