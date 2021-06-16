# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer

from hypernets.tabular import sklearn_ex as skex, dask_ex as dex


def select_by_multicollinearity(X, method=None):
    """
    Adapted from https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    handling multicollinearity is by performing hierarchical clustering on the features’ Spearman
    rank-order correlations, picking a threshold, and keeping a single feature from each cluster.

    :param X:
    :return:
    """
    if (method is None or method == 'spearman') and isinstance(X, pd.DataFrame):
        Xt = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(X)
        corr = spearmanr(Xt).correlation
    elif isinstance(X, pd.DataFrame):
        Xt = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(X)
        Xt = skex.SafeOrdinalEncoder().fit_transform(Xt)
        corr = Xt.corr(method=method).values
    else:  # dask
        Xt = dex.SafeOrdinalEncoder().fit_transform(X)
        corr = Xt.corr(method='pearson' if method is None else method).compute().values

    corr_linkage = hierarchy.ward(corr)

    cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected = [X.columns[v[0]] for v in cluster_id_to_feature_ids.values()]
    unselected = list(set(X.columns.to_list()) - set(selected))
    feature_clusters = [[X.columns[i] for i in v] for v in cluster_id_to_feature_ids.values()]
    return feature_clusters, selected, unselected
