# -*- coding:utf-8 -*-
"""

"""
import cudf
import cupy
import numpy as np
from cuml import model_selection as cm_sel
from cuml.preprocessing import LabelEncoder
from sklearn import model_selection as sk_sel

from hypernets.utils import logging

logger = logging.get_logger(__name__)


def train_test_split(X, y=None, shuffle=True, random_state=None, stratify=None, **kwargs):
    if y is not None and str(y.dtype) == 'object':
        return _train_test_split_with_object(X, y, shuffle=shuffle, random_state=random_state, stratify=stratify,
                                             **kwargs)
    try:
        return cm_sel.train_test_split(X, y, shuffle=shuffle, random_state=random_state, stratify=stratify,
                                       **kwargs)
    except Exception as e:
        if stratify is not None and str(e).find('cudaErrorInvalidValue') >= 0:
            logger.warning('train_test_split failed, retry without stratify')
            return cm_sel.train_test_split(X, y, shuffle=shuffle, random_state=random_state, **kwargs)
        else:
            raise


def _train_test_split_with_object(X, y, shuffle=True, random_state=None, stratify=None, **kwargs):
    """
    cuml.train_test_split raise exception if y.dtype=='object', so we encode it
    """
    le = LabelEncoder()
    yt = le.fit_transform(y)

    if stratify is y:
        stratify = yt
    elif stratify is not None and str(stratify.dtype) == 'object':
        stratify = LabelEncoder().fit_transform(stratify)

    X_train, X_test, y_train, y_test = \
        cm_sel.train_test_split(X, yt, shuffle=shuffle, random_state=random_state, stratify=stratify, **kwargs)

    y_train_decoded = le.inverse_transform(y_train)
    y_test_decoded = le.inverse_transform(y_test)
    y_train_decoded.index = y_train.index
    y_test_decoded.index = y_test.index

    return X_train, X_test, y_train_decoded, y_test_decoded


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
