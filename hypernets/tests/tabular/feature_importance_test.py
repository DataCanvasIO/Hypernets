# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from collections import defaultdict

import numpy as np
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from hypernets.core import set_random_state, get_random_state
from hypernets.examples.plain_model import train
from hypernets.tabular import get_tool_box
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.sklearn_ex import MultiLabelEncoder


def test_collinear():
    df = dsutils.load_bank().head(10000)
    y = df.pop('y')
    df.drop(['id'], axis=1, inplace=True)
    corr = spearmanr(df).correlation
    corr_linkage = hierarchy.ward(corr)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    # dendro = hierarchy.dendrogram(
    #     corr_linkage, labels=df.columns.to_list(), ax=ax1, leaf_rotation=90
    # )
    # dendro_idx = np.arange(0, len(dendro['ivl']))
    # ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    # ax2.set_xticks(dendro_idx)
    # ax2.set_yticks(dendro_idx)
    # ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    # ax2.set_yticklabels(dendro['ivl'])
    # fig.tight_layout()
    # plt.show()

    cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [df.columns[v[0]] for v in cluster_id_to_feature_ids.values()]
    assert selected_features == ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                                 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'poutcome']


class TestPermutationImportance:
    @staticmethod
    def load_data():
        set_random_state(9527)
        df = dsutils.load_bank().head(3000)
        encoder = MultiLabelEncoder()
        df = encoder.fit_transform(df)
        df.drop(['id'], axis=1, inplace=True)
        return df

    def test_basic(self):
        df = self.load_data()
        y = df.pop('y')
        tb = get_tool_box(df)

        X_train, X_test, y_train, y_test = tb.train_test_split(df, y, test_size=0.3, random_state=42)

        hm, _ = train(X_train, y_train, X_test, y_test, max_trials=5)

        best_trials = hm.get_top_trials(3)
        estimators = [hm.load_estimator(trial.model_file) for trial in best_trials]

        importances = tb.permutation_importance_batch(estimators, X_test, y_test, get_scorer('roc_auc_ovr'), n_jobs=1,
                                                      n_repeats=5, random_state=get_random_state())

        feature_index = np.argwhere(importances.importances_mean < 1e-5)
        selected_features = [feat for i, feat in enumerate(X_train.columns.to_list()) if i not in feature_index]
        unselected_features = [c for c in X_train.columns.to_list() if c not in selected_features]
        set_random_state(None)

        print('selected:  ', selected_features)
        print('unselected:', unselected_features)
        # assert selected_features == ['job', 'marital', 'education', 'balance', 'housing', 'loan', 'contact', 'day',
        #                              'duration', 'campaign', 'pdays', 'previous', 'poutcome']
        # assert unselected_features == ['age', 'default', 'month']
