# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from hypernets.tabular.datasets.dsutils import load_bank
from hypernets.tabular.drift_detection import FeatureSelectorWithDriftDetection, DriftDetector

matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)


class Test_drift_detection:
    def test_shift_score(self):
        df = load_bank().head(1000)
        selector = FeatureSelectorWithDriftDetection()
        scores = selector._covariate_shift_score(df[:700], df[700:], shuffle=False)
        assert scores['id'] == 1.0

    def test_shift_score_with_matthews_corrcoef(self):
        df = load_bank().head(1000)
        selector = FeatureSelectorWithDriftDetection()
        scores = selector._covariate_shift_score(df[:700], df[700:], scorer=matthews_corrcoef_scorer, shuffle=False)
        assert scores['id'] == 1.0

    def test_shift_score_cv(self):
        df = load_bank().head(1000)
        selector = FeatureSelectorWithDriftDetection()
        scores = selector._covariate_shift_score(df[:700], df[700:], cv=5, shuffle=False)
        assert scores['id'] >= 0.95

    def test_shufflesplit(self):
        df = load_bank().head(1000)
        y = df.pop('y')
        iterators = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
        indices = []
        for train_index, test_index in iterators.split(df, y):
            indices.append((train_index, test_index))
        assert len(indices) == 1
        assert len(indices[0][0]) == 700
        assert len(indices[0][1]) == 300

    def test_drift_detector_lightgbm(self):
        df = load_bank()
        y = df.pop('y')
        X_train, X_test = train_test_split(df.copy(), train_size=0.7, shuffle=True, random_state=9527)
        dd = DriftDetector()
        dd.fit(X_train, X_test)

        assert len(dd.feature_names_) == 17
        assert len(dd.feature_importances_) == 17
        assert dd.auc_
        assert len(dd.estimator_) == 5

        proba = dd.predict_proba(df)
        assert proba.shape[0] == df.shape[0]

        df = load_bank()
        y = df.pop('y')
        p = int(df.shape[0] * 0.2)
        X_train, X_test, y_train, y_test = dd.train_test_split(df.copy(), y, test_size=0.2)
        assert X_train.shape == (df.shape[0] - p, df.shape[1])
        assert y_train.shape == (df.shape[0] - p,)
        assert X_test.shape == (p, df.shape[1])
        assert y_test.shape == (p,)

        df['y'] = y
        X_train['y'] = y_train
        X_test['y'] = y_test
        df_split = pd.concat([X_train, X_test])
        df_hash = hash_pandas_object(df).sort_values()
        splitted_hash = hash_pandas_object(df_split).sort_values()
        assert (df_hash == splitted_hash).all()

    def test_drift_detector_fit_decisiontree(self):
        df = load_bank().head(10000)
        y = df.pop('y')
        X_train, X_test = train_test_split(df, train_size=0.7, shuffle=True, random_state=9527)

        dd_dt = DriftDetector(
            estimator=DecisionTreeClassifier(min_samples_leaf=20, min_impurity_decrease=0.01))

        dd_dt.fit(X_train, X_test)

        assert len(dd_dt.feature_names_) == 17
        assert len(dd_dt.feature_importances_) == 17
        assert dd_dt.auc_
        assert len(dd_dt.estimator_) == 5

    def test_drift_detector_fit_randomforest(self):
        df = load_bank().head(10000)
        y = df.pop('y')
        X_train, X_test = train_test_split(df, train_size=0.7, shuffle=True, random_state=9527)

        dd_rf = DriftDetector(
            estimator=RandomForestClassifier(min_samples_leaf=20, min_impurity_decrease=0.01))

        dd_rf.fit(X_train, X_test)

        assert len(dd_rf.feature_names_) == 17
        assert len(dd_rf.feature_importances_) == 17
        assert dd_rf.auc_
        assert len(dd_rf.estimator_) == 5

    def test_feature_selection(self):
        df = load_bank()
        y = df.pop('y')
        p = int(df.shape[0] * 0.8)
        X_train = df[:p]
        X_test = df[p:]
        # = train_test_split(df, train_size=0.7,  random_state=9527)
        selector = FeatureSelectorWithDriftDetection(remove_shift_variable=False,
                                                     auc_threshold=0.55,
                                                     min_features=15,
                                                     remove_size=0.2)
        remain_features, history, scores = selector.select(X_train, X_test, copy_data=True)
        assert len(remain_features) == 15

        selector = FeatureSelectorWithDriftDetection(remove_shift_variable=True,
                                                     auc_threshold=0.55,
                                                     min_features=15,
                                                     remove_size=0.2)
        remain_features, history, scores = selector.select(X_train, X_test, copy_data=True)

        assert len(remain_features) in [15, 16]
