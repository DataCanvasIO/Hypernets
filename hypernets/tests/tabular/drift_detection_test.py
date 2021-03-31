# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from hypernets.tabular.datasets.dsutils import load_bank
from hypernets.tabular.drift_detection import covariate_shift_score, DriftDetector, feature_selection

matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)


class Test_drift_detection:
    def test_shift_score(self):
        df = load_bank().head(1000)
        scores = covariate_shift_score(df[:700], df[700:])
        assert scores['id'] == 1.0

    def test_shift_score_with_matthews_corrcoef(self):
        df = load_bank().head(1000)
        scores = covariate_shift_score(df[:700], df[700:], scorer=matthews_corrcoef_scorer)
        assert scores['id'] == 1.0

    def test_shift_score_cv(self):
        df = load_bank().head(1000)
        scores = covariate_shift_score(df[:700], df[700:], cv=5)
        assert scores['id'] == 0.95

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
        df = load_bank().head(10000)
        y = df.pop('y')
        X_train, X_test = train_test_split(df, train_size=0.7, shuffle=True, random_state=9527)
        dd = DriftDetector()
        dd.fit(X_train, X_test)

        assert len(dd.feature_names_) == 17
        assert len(dd.feature_importances_) == 17
        assert dd.auc_
        assert len(dd.estimator_) == 5

        proba = dd.predict_proba(df)
        assert proba.shape[0] == 10000

        df = load_bank()
        y = df.pop('y')
        X_train, X_test, y_train, y_test = dd.train_test_split(df, y, test_size=0.2)
        assert X_train.shape, (86804, 17)
        assert y_train.shape, (86804,)
        assert X_test.shape, (21700, 17)
        assert y_test.shape, (21700,)

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
        df = load_bank().head(10000)
        y = df.pop('y')
        X_train = df[:7000]
        X_test = df[7000:]
        # = train_test_split(df, train_size=0.7,  random_state=9527)

        remain_features, history, scores = feature_selection(X_train, X_test, remove_shift_variable=False, auc_threshold=0.55,
                                                     min_features=15,
                                                     remove_size=0.2, copy_data=True)
        assert len(remain_features) == 15

        remain_features, history, scores = feature_selection(X_train, X_test, remove_shift_variable=True, auc_threshold=0.55,
                                                     min_features=15,
                                                     remove_size=0.2, copy_data=True)

        assert len(remain_features) == 16
