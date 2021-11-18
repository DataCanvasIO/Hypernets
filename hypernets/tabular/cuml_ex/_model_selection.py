# -*- coding:utf-8 -*-
"""

"""
import cudf
import cupy
import numpy as np
from cuml import model_selection as cm_sel
from sklearn import model_selection as sk_sel

from hypernets.utils import logging

logger = logging.get_logger(__name__)


def train_test_split(*data, shuffle=True, random_state=None, stratify=None, **kwargs):
    try:
        return cm_sel.train_test_split(*data, shuffle=shuffle, random_state=random_state, stratify=stratify, **kwargs)
    except Exception as e:
        if stratify is not None and str(e).find('cudaErrorInvalidValue') >= 0:
            logger.warning('train_test_split failed, retry without stratify')
            return cm_sel.train_test_split(*data, shuffle=shuffle, random_state=random_state, **kwargs)
        else:
            raise e


class FakeKFold(sk_sel.KFold):
    def split(self, X, y=None, groups=None):
        assert y is None or len(X) == len(y)
        X = y = np.arange(len(X))

        yield from super().split(X, y, groups=groups)


class FakeStratifiedKFold(sk_sel.StratifiedKFold):
    def split(self, X, y, groups=None):
        assert y is not None

        X = np.arange(len(X))

        if isinstance(y, (cudf.Series, cudf.DataFrame)):
            y = y.to_pandas()
        elif isinstance(y, cupy.ndarray):
            y = cupy.asnumpy(y)

        yield from super().split(X, y, groups=groups)
