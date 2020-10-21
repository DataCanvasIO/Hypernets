# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from lightgbm.sklearn import LGBMClassifier
from copy import deepcopy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from ..column_selector import column_object_category_bool, column_number_exclude_timedelta


def covariate_shift_score(X_train, X_test, copy=True):
    assert isinstance(X_train, pd.DataFrame) and isinstance(X_test,
                                                            pd.DataFrame), 'X_train and X_test must be a pandas DataFrame.'
    assert len(set(X_train.columns.to_list()) - set(
        X_test.columns.to_list())) == 0, 'The columns in X_train and X_test must be the same.'
    target_col = '__hypernets_csd__target__'
    if copy:
        train = deepcopy(X_train)
        test = deepcopy(X_test)
    else:
        train = X_train
        test = X_test

    #Set target value
    train[target_col] = 0
    test[target_col] = 1
    mixed = pd.concat([train, test], axis=0)
    y = mixed.pop(target_col)

    #Preprocess data: imputing and scaling
    cat_cols = column_object_category_bool(mixed)
    num_cols = column_number_exclude_timedelta(mixed)
    cat_transformer = Pipeline(
        steps=[('imputer_cat', SimpleImputer(strategy='constant')), ('encoder', OrdinalEncoder())])
    num_transformer = Pipeline(steps=[('imputer_num', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('cat', cat_transformer, cat_cols),
                                                   ('num', num_transformer, num_cols)],
                                     remainder='passthrough')
    mixed[cat_cols + num_cols] = preprocessor.fit_transform(mixed)

    #Calculate the shift score for each column separately.
    scores = {}
    for c in mixed.columns:
        x = mixed[[c]]
        mixed_x_train, mixed_x_test, mixed_y_train, mixed_y_test = train_test_split(x, y, test_size=0.3,
                                                                                    random_state=9527, stratify=y)

        classifier = LGBMClassifier()
        classifier.fit(mixed_x_train, mixed_y_train, eval_set=(mixed_x_test, mixed_y_test), early_stopping_rounds=20)
        y_pred = classifier.predict_proba(mixed_x_test)
        auc = roc_auc_score(mixed_y_test, y_pred[:, 1])
        scores[c] = auc

    return scores
