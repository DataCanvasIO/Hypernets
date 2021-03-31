# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold
from hypernets.utils import logging

logger = logging.get_logger(__name__)


def select_valid_oof(y, oof):
    if len(oof.shape) == 1:
        idx = np.argwhere(~np.isnan(oof[:])).ravel()
    elif len(oof.shape) == 2:
        idx = np.argwhere(~np.isnan(oof[:, 0])).ravel()
    elif len(oof.shape) == 3:
        idx = np.argwhere(~np.isnan(oof[:, 0, 0])).ravel()
    else:
        raise ValueError(f'Unsupported shape:{oof.shape}')
    return y.iloc[idx] if hasattr(y, 'iloc') else y[idx], oof[idx]


class PrequentialSplit(_BaseKFold):
    STRATEGY_PREQ_BLS = 'preq-bls'
    STRATEGY_PREQ_SLID_BLS = 'preq-slid-bls'
    STRATEGY_PREQ_BLS_GAP = 'preq-bls-gap'
    """

    Parameters
    ----------
        mode : Strategies of requential approach applied in blocks for performance estimation
            `preq-bls`:
            `preq-slid-bls`:
            `preq-bls-gap`:

    References
    ----------
        Cerqueira V, Torgo L, Mozetič I. Evaluating time series forecasting models: An empirical study on performance estimation methods[J]. Machine Learning, 2020, 109(11): 1997-2028.
    """

    def __init__(self, strategy='preq-bls', base_size=None, n_splits=5, stride=1, *, max_train_size=None):
        super(PrequentialSplit, self).__init__(n_splits=max((n_splits // stride) - 1, 2), shuffle=False,
                                               random_state=None)
        self.max_train_size = max_train_size
        self.base_size = base_size
        self.stride = stride
        self.n_folds = n_splits
        self.strategy = strategy
        self.fold_size = None

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        base = 0
        if self.base_size is not None and self.base_size > 0:
            base = self.base_size
        base += n_samples % self.n_folds

        if self.n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(self.n_folds,
                                                             n_samples))

        indices = np.arange(n_samples)
        fold_size = (n_samples - base) // self.n_folds
        self.fold_size = fold_size
        logger.info(f'n_folds:{self.n_folds}')
        logger.info(f'fold_size:{fold_size}')
        if self.strategy == PrequentialSplit.STRATEGY_PREQ_BLS_GAP:
            test_starts = range(fold_size * 2 + base, n_samples, fold_size)
        else:
            test_starts = range(fold_size + base, n_samples, fold_size)
        last_step = -1
        for fold, test_start in enumerate(test_starts):
            if last_step == fold // self.stride:
                # skip this fold
                continue
            else:
                last_step = fold // self.stride
            if self.strategy == PrequentialSplit.STRATEGY_PREQ_BLS:
                yield (indices[:test_start], indices[test_start:test_start + fold_size])
            elif self.strategy == PrequentialSplit.STRATEGY_PREQ_SLID_BLS:
                if self.max_train_size and self.max_train_size < test_start:
                    yield (indices[test_start - self.max_train_size:test_start],
                           indices[test_start:test_start + fold_size])
                else:
                    yield (indices[test_start - (fold_size + base):test_start],
                           indices[test_start:test_start + fold_size])
            elif self.strategy == PrequentialSplit.STRATEGY_PREQ_BLS_GAP:
                yield (indices[:test_start - fold_size], indices[test_start:test_start + fold_size])
            else:
                raise ValueError(f'{self.strategy} is not supported')
