# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import pytest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, get_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier

from hypernets.tabular.column_selector import column_object_category_bool, column_number_exclude_timedelta
from hypernets.tabular.dataframe_mapper import DataFrameMapper
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.ensemble import StackingEnsemble, AveragingEnsemble, GreedyEnsemble


def general_preprocessor():
    cat_transformer = Pipeline(
        steps=[('imputer_cat', SimpleImputer(strategy='constant')), ('encoder', OrdinalEncoder())])
    num_transformer = Pipeline(steps=[('imputer_num', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])

    preprocessor = DataFrameMapper(features=[(column_object_category_bool, cat_transformer),
                                             (column_number_exclude_timedelta, num_transformer)],
                                   input_df=True,
                                   df_out=True)
    return preprocessor


class TestStacking():
    def test_scorer(self):
        scorer = get_scorer('neg_log_loss')
        assert scorer

    def test_multi_estimators_binary(self):
        preprocessor = general_preprocessor()
        df = dsutils.load_bank().head(2000)
        y = df.pop('y')
        X = preprocessor.fit_transform(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=9527)

        estimators = [RandomForestClassifier(), DecisionTreeClassifier(), ExtraTreeClassifier(), LogisticRegression(),
                      GradientBoostingClassifier()]

        rf_auc = self.get_auc(estimators[0], X_train, X_test, y_train, y_test)
        dt_auc = self.get_auc(estimators[1], X_train, X_test, y_train, y_test)
        et_auc = self.get_auc(estimators[2], X_train, X_test, y_train, y_test)
        lr_auc = self.get_auc(estimators[3], X_train, X_test, y_train, y_test)
        gb_auc = self.get_auc(estimators[4], X_train, X_test, y_train, y_test)

        ests = [RandomForestClassifier(), DecisionTreeClassifier(), LogisticRegression(), ExtraTreeClassifier(),
                LogisticRegression(), GradientBoostingClassifier()]

        greedy = GreedyEnsemble('binary', ests, need_fit=True, n_folds=5, ensemble_size=10)
        greedy_auc = self.get_auc(greedy, X_train, X_test, y_train, y_test)

        # save and load
        greedy.save('test_ensemble_output_')
        greedy = GreedyEnsemble.load('test_ensemble_output_')

        pred = greedy.predict(X_test)

        greedy = GreedyEnsemble('binary', ests, need_fit=True, n_folds=5, ensemble_size=10, scoring='accuracy')
        greedy_auc2 = self.get_auc(greedy, X_train, X_test, y_train, y_test)

        greedy = GreedyEnsemble('binary', ests, need_fit=True, n_folds=5, ensemble_size=10, scoring='roc_auc_ovo')
        greedy_auc3 = self.get_auc(greedy, X_train, X_test, y_train, y_test)

        avg = AveragingEnsemble('binary', ests, need_fit=True, n_folds=5, )
        avg_auc = self.get_auc(avg, X_train, X_test, y_train, y_test)

        avg_hard = AveragingEnsemble('binary', ests, need_fit=True, n_folds=5, method='hard')
        avg_auc_hard = self.get_auc(avg_hard, X_train, X_test, y_train, y_test)

        stacking_hard = StackingEnsemble('binary', ests, need_fit=True, n_folds=5, method='hard')
        stacking_auc_hard = self.get_auc(stacking_hard, X_train, X_test, y_train, y_test)

        stacking_soft = StackingEnsemble('binary', ests, need_fit=True, n_folds=5, method='soft')
        stacking_auc_soft = self.get_auc(stacking_soft, X_train, X_test, y_train, y_test)

        assert stacking_auc_soft

    def test_multi_estimators_multiclass(self):
        preprocessor = general_preprocessor()
        df = dsutils.load_glass_uci()
        y = df.pop(10)
        X = preprocessor.fit_transform(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=9527)

        estimators = [RandomForestClassifier(), DecisionTreeClassifier(), ExtraTreeClassifier(), LogisticRegression(),
                      GradientBoostingClassifier()]

        rf_auc = self.get_auc(estimators[0], X_train, X_test, y_train, y_test)
        dt_auc = self.get_auc(estimators[1], X_train, X_test, y_train, y_test)
        et_auc = self.get_auc(estimators[2], X_train, X_test, y_train, y_test)
        lr_auc = self.get_auc(estimators[3], X_train, X_test, y_train, y_test)
        gb_auc = self.get_auc(estimators[4], X_train, X_test, y_train, y_test)

        ests = [RandomForestClassifier(), DecisionTreeClassifier(), ExtraTreeClassifier(), LogisticRegression(),
                GradientBoostingClassifier()]

        greedy = GreedyEnsemble('multiclass', ests, need_fit=True, n_folds=5, ensemble_size=10)
        greedy_auc = self.get_auc(greedy, X_train, X_test, y_train, y_test)

        avg = AveragingEnsemble('multiclass', ests, need_fit=True, n_folds=5, )
        avg_auc = self.get_auc(avg, X_train, X_test, y_train, y_test)

        avg_hard = AveragingEnsemble('multiclass', ests, need_fit=True, n_folds=5, method='hard')
        with pytest.raises(ValueError) as err:
            avg_auc_hard = self.get_auc(avg_hard, X_train, X_test, y_train, y_test)
            assert err

        stacking_hard = StackingEnsemble('multiclass', ests, need_fit=True, n_folds=5, method='hard')
        stacking_auc_hard = self.get_auc(stacking_hard, X_train, X_test, y_train, y_test)

        stacking_soft = StackingEnsemble('multiclass', ests, need_fit=True, n_folds=5, method='soft')
        stacking_auc_soft = self.get_auc(stacking_soft, X_train, X_test, y_train, y_test)

        assert stacking_auc_soft

    def get_auc(self, clf, X_train, X_test, y_train, y_test):
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        if proba.shape[1] == 2:
            pred = proba[:, 1]
        else:
            pred = proba
        auc = roc_auc_score(y_test, pred, multi_class='ovo')
        return auc
