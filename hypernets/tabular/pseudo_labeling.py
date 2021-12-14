# -*- coding:utf-8 -*-
"""

"""

from hypernets.utils import logging

logger = logging.get_logger(__name__)


class PseudoLabeling:
    DEFAULT_STRATEGY_SETTINGS = dict(
        default_strategy='threshold',
        default_threshold=0.8,
        default_quantile=0.8,
        default_number=0.2,
    )
    import numpy as np

    def __init__(self, strategy, threshold=None, quantile=None, number=None):
        strategy, threshold, quantile, number = \
            self.detect_strategy(strategy, threshold=threshold, quantile=quantile, number=number)

        self.strategy = strategy
        self.threshold = threshold
        self.quantile = quantile
        self.number = number

    @staticmethod
    def detect_strategy(strategy, threshold=None, quantile=None, number=None):
        from .toolbox import ToolBox
        return ToolBox.detect_strategy(strategy, threshold=threshold, quantile=quantile, number=number,
                                       **PseudoLabeling.DEFAULT_STRATEGY_SETTINGS)

    def select(self, X_test, classes, proba):
        assert len(classes) == proba.shape[-1] > 1
        from . import ToolBox, get_tool_box

        np = self.np

        proba = np.array(proba)
        mx = proba.max(axis=1, keepdims=True)
        proba = np.where(proba < mx, 0, proba)

        if self.strategy is None or self.strategy == ToolBox.STRATEGY_THRESHOLD:
            selected = self._filter_by_threshold(proba)
        elif self.strategy == ToolBox.STRATEGY_NUMBER:
            selected = self._filter_by_number(proba)
        elif self.strategy == ToolBox.STRATEGY_QUANTILE:
            selected = self._filter_by_quantile(proba)
        else:
            raise ValueError(f'Unsupported strategy: {self.strategy}')

        pred = (selected * np.arange(1, len(classes) + 1)).max(axis=1) - 1
        idx = np.argwhere(pred >= 0).ravel()

        # X_pseudo = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
        # y_pseudo = np.take(np.array(classes), pred[idx], axis=0)
        tb = get_tool_box(X_test)
        X_pseudo = tb.select_1d(X_test, idx)
        y_pseudo = tb.take_array(classes, pred[idx], axis=0)

        if logger.is_info_enabled():
            msg_prefix = f'[{type(self).__name__}] extract pseudo labeling samples (strategy={self.strategy})'
            if len(y_pseudo) > 0:
                value_counts = tb.value_counts(y_pseudo)
                logger.info(f'{msg_prefix}: {value_counts}')
            else:
                logger.info(f'{msg_prefix}: nothing')
        return X_pseudo, y_pseudo

    def _filter_by_threshold(self, proba):
        selected = (proba >= self.threshold)
        return selected

    def _filter_by_number(self, proba):
        np = self.np
        if isinstance(self.number, float) and 0 < self.number < 1:
            number = int(proba.shape[0] / proba.shape[1] * self.number)
            if number < 10:
                number = 10
        else:
            number = int(self.number)

        pos = proba.shape[0] - number
        i = np.argsort(np.argsort(proba, axis=0), axis=0)
        selected = np.logical_and(i >= pos, proba > 0)
        return selected

    def _filter_by_quantile(self, proba):
        np = self.np
        qs = np.nanquantile(np.where(proba > 0, proba, np.nan), self.quantile, axis=0)
        selected = (proba >= qs)
        return selected
