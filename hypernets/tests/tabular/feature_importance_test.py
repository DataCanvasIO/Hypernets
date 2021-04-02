# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from collections import defaultdict

from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from hypernets.examples.plain_model import train
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.feature_importance import feature_importance_batch
from hypernets.tabular.sklearn_ex import MultiLabelEncoder


class Test_FeatureImportance():

    def test_collinear(self):
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

    def test_basic(self):
        df = dsutils.load_bank().head(100)
        encoder = MultiLabelEncoder()
        df = encoder.fit_transform(df)
        y = df.pop('y')
        df.drop(['id'], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

        hm, _ = train(X_train, y_train, X_test, y_test, max_trials=10)

        best_trials = hm.get_top_trials(3)
        estimators = [hm.load_estimator(trial.model_file) for trial in best_trials]

        importances = feature_importance_batch(estimators, X_test, y_test, get_scorer('roc_auc_ovr'), n_repeats=2)
        assert importances
