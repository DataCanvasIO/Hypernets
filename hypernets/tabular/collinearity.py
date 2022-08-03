# -*- coding:utf-8 -*-
"""
Adapted from https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
handling multicollinearity is by performing hierarchical clustering on the features’ Spearman
rank-order correlations, picking a threshold, and keeping a single feature from each cluster.
"""

from collections import defaultdict

import numpy as np
from scipy.cluster import hierarchy
from scipy.stats import spearmanr

from hypernets.core import randint
from hypernets.tabular import sklearn_ex as skex
from hypernets.tabular.cfg import TabularCfg as cfg
from hypernets.utils import logging

logger = logging.get_logger(__name__)


class MultiCollinearityDetector:
    def detect(self, X, method=None):
        from . import get_tool_box

        tb = get_tool_box(X)
        X_shape = tb.get_shape(X)
        sample_limit = cfg.multi_collinearity_sample_limit
        if X_shape[0] > sample_limit:
            logger.info(f'{X_shape[0]} rows data found, sample to {sample_limit}')
            frac = sample_limit / X_shape[0]
            X, _, = tb.train_test_split(X, train_size=frac, random_state=randint())

        n_values = self._value_counts(X)
        one_values = [n.name for n in n_values if len(n) <= 1]
        if len(one_values) > 0:
            X = X[[c for c in X.columns if c not in one_values]]

        logger.info('computing correlation')
        corr = self._corr(X, method)

        logger.info('computing cluster')
        corr_linkage = hierarchy.ward(corr)
        cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected = [X.columns[v[0]] for v in cluster_id_to_feature_ids.values()]
        unselected = list(set(X.columns.to_list()) - set(selected)) + one_values
        feature_clusters = [[X.columns[i] for i in v] for v in cluster_id_to_feature_ids.values()]

        return feature_clusters, selected, unselected

    def _value_counts(self, X):
        return [X[c].value_counts() for c in X.columns]

    def _corr(self, X, method=None):
        if method is None or method == 'spearman':
            Xt = skex.SafeSimpleImputer(missing_values=np.nan, strategy='most_frequent') \
                .fit_transform(X)
            corr = spearmanr(Xt).correlation
        else:
            from . import get_tool_box
            Xt = X.copy()
            cols = get_tool_box(X).column_selector.column_number_exclude_timedelta(X)
            if cols:
                Xt[cols] = skex.SafeSimpleImputer(missing_values=np.nan, strategy='most_frequent') \
                    .fit_transform(Xt[cols])
            Xt = skex.SafeOrdinalEncoder().fit_transform(Xt)
            corr = Xt.corr(method=method).values

        return corr
