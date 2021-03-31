# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.tabular.lifelong_learning import PrequentialSplit
import numpy as np


class Test_PrequentialSplit():

    def get_train_test_index(self, split, X):
        train = []
        test = []
        for train_indices, test_indices in split.split(X):
            train.append((train_indices[0], train_indices[-1]))
            test.append((test_indices[0], test_indices[-1]))
        return train, test

    def test_strategy_preq_bls(self):
        X = np.zeros((1003, 4))

        split = PrequentialSplit(strategy=PrequentialSplit.STRATEGY_PREQ_BLS, base_size=None, n_splits=5,
                                 max_train_size=200)
        train, test = self.get_train_test_index(split, X)
        assert split.fold_size == 200
        assert train == [(0, 202), (0, 402), (0, 602), (0, 802)]
        assert test == [(203, 402), (403, 602), (603, 802), (803, 1002)]

        split = PrequentialSplit(strategy=PrequentialSplit.STRATEGY_PREQ_BLS, base_size=50, n_splits=5,
                                 max_train_size=200)
        train, test = self.get_train_test_index(split, X)
        assert split.fold_size == 190
        assert train == [(0, 242), (0, 432), (0, 622), (0, 812)]
        assert test == [(243, 432), (433, 622), (623, 812), (813, 1002)]

        split = PrequentialSplit(strategy=PrequentialSplit.STRATEGY_PREQ_BLS, base_size=50, n_splits=5,
                                 stride=3,
                                 max_train_size=200)
        train, test = self.get_train_test_index(split, X)
        assert split.fold_size == 190
        assert train == [(0, 242), (0, 812)]
        assert test == [(243, 432), (813, 1002)]

    def test_strategy_preq_slid_bls(self):
        X = np.zeros((1003, 4))

        split = PrequentialSplit(strategy=PrequentialSplit.STRATEGY_PREQ_SLID_BLS, base_size=None, n_splits=5,
                                 max_train_size=200)
        train, test = self.get_train_test_index(split, X)
        assert split.fold_size == 200
        assert train == [(3, 202), (203, 402), (403, 602), (603, 802)]
        assert test == [(203, 402), (403, 602), (603, 802), (803, 1002)]

        split = PrequentialSplit(strategy=PrequentialSplit.STRATEGY_PREQ_SLID_BLS, base_size=None, n_splits=5,
                                 max_train_size=500)
        train, test = self.get_train_test_index(split, X)
        assert split.fold_size == 200
        assert train == [(0, 202), (200, 402), (103, 602), (303, 802)]
        assert test == [(203, 402), (403, 602), (603, 802), (803, 1002)]

        split = PrequentialSplit(strategy=PrequentialSplit.STRATEGY_PREQ_SLID_BLS, base_size=50, n_splits=5,
                                 max_train_size=500)
        train, test = self.get_train_test_index(split, X)
        assert split.fold_size == 190
        assert train == [(0, 242), (190, 432), (123, 622), (313, 812)]
        assert test == [(243, 432), (433, 622), (623, 812), (813, 1002)]

        split = PrequentialSplit(strategy=PrequentialSplit.STRATEGY_PREQ_SLID_BLS, base_size=50, n_splits=5,
                                 stride=3, max_train_size=500)
        train, test = self.get_train_test_index(split, X)
        assert split.fold_size == 190
        assert train == [(0, 242), (313, 812)]
        assert test == [(243, 432), (813, 1002)]

    def test_strategy_preq_bls_gap(self):
        X = np.zeros((1003, 4))

        split = PrequentialSplit(strategy=PrequentialSplit.STRATEGY_PREQ_BLS_GAP, base_size=None, n_splits=5,
                                 max_train_size=200)
        train, test = self.get_train_test_index(split, X)
        assert split.fold_size == 200
        assert train == [(0, 202), (0, 402), (0, 602)]
        assert test == [(403, 602), (603, 802), (803, 1002)]

        split = PrequentialSplit(strategy=PrequentialSplit.STRATEGY_PREQ_BLS_GAP, base_size=50, n_splits=5,
                                 max_train_size=200)
        train, test = self.get_train_test_index(split, X)
        assert split.fold_size == 190
        assert train == [(0, 242), (0, 432), (0, 622)]
        assert test == [(433, 622), (623, 812), (813, 1002)]

        split = PrequentialSplit(strategy=PrequentialSplit.STRATEGY_PREQ_BLS_GAP, base_size=50, n_splits=5,
                                 stride=2, max_train_size=200)
        train, test = self.get_train_test_index(split, X)
        assert split.fold_size == 190
        assert train == [(0, 242), (0, 622)]
        assert test == [(433, 622), (813, 1002)]
