# -*- coding:utf-8 -*-
"""

"""

from dask import dataframe as dd

from hypernets.utils import const, logging
from ..metrics import Metrics

logger = logging.get_logger(__name__)

_DASK_METRICS = ('accuracy', 'logloss')


def _calc_score_dask(y_true, y_preds, y_proba=None, metrics=('accuracy',), task=const.TASK_BINARY, pos_label=1,
                     classes=None, average=None):
    import dask_ml.metrics as dm_metrics
    from ._toolbox import DaskToolBox

    def to_array(name, value):
        if value is None:
            return value

        if isinstance(value, (dd.DataFrame, dd.Series)):
            value = value.values

        if len(value.shape) == 2 and value.shape[-1] == 1:
            value = value.reshape(-1)

        value = DaskToolBox.make_chunk_size_known(value)
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


class DaskMetrics(Metrics):
    @staticmethod
    def calc_score(y_true, y_preds, y_proba=None, metrics=('accuracy',), task=const.TASK_BINARY, pos_label=1,
                   classes=None, average=None):
        from ._toolbox import DaskToolBox

        data_list = (y_true, y_proba, y_preds)
        if any(map(DaskToolBox.is_dask_object, data_list)):
            if all(map(DaskToolBox.is_dask_object, data_list)) and len(
                    set(metrics).difference(set(_DASK_METRICS))) == 0:
                fn = _calc_score_dask
            else:
                y_true, y_proba, y_preds = \
                    [y.compute() if DaskToolBox.is_dask_object(y) else y for y in data_list]
                fn = Metrics.calc_score
        else:
            fn = Metrics.calc_score

        return fn(y_true, y_preds, y_proba, metrics, task=task, pos_label=pos_label,
                  classes=classes, average=average)
