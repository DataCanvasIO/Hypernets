# -*- coding:utf-8 -*-
"""

"""
import copy

import lightgbm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from hypernets.tabular import sklearn_ex as skex, dask_ex as dex
from hypernets.tabular.column_selector import column_object_category_bool, column_number_exclude_timedelta
from hypernets.tabular.dataframe_mapper import DataFrameMapper
from hypernets.utils import const


def general_preprocessor(X):
    if dex.is_dask_dataframe(X):
        import dask_ml.impute as dimp
        import dask_ml.preprocessing as dpre
        cat_steps = [('imputer_cat', dimp.SimpleImputer(strategy='constant', fill_value='')),
                     ('encoder', dex.SafeOrdinalEncoder())]
        num_steps = [('imputer_num', dimp.SimpleImputer(strategy='mean')),
                     ('scaler', dpre.StandardScaler())]
    else:
        cat_steps = [('imputer_cat', SimpleImputer(strategy='constant', fill_value='')),
                     ('encoder', skex.SafeOrdinalEncoder())]
        num_steps = [('imputer_num', SimpleImputer(strategy='mean')),
                     ('scaler', StandardScaler())]

    cat_transformer = Pipeline(steps=cat_steps)
    num_transformer = Pipeline(steps=num_steps)

    preprocessor = DataFrameMapper(features=[(column_object_category_bool, cat_transformer),
                                             (column_number_exclude_timedelta, num_transformer)],
                                   input_df=True,
                                   df_out=True)
    return preprocessor


def _wrap_predict_proba(estimator):
    orig_predict_proba = estimator.predict_proba

    def __predict_proba(*args, **kwargs):
        proba = orig_predict_proba(*args, **kwargs)
        proba = dex.fix_binary_predict_proba_result(proba)
        return proba

    setattr(estimator, '_orig_predict_proba', orig_predict_proba)
    setattr(estimator, 'predict_proba', __predict_proba)
    return estimator


def general_estimator(X, estimator=None, task=None):
    def default_gbm():
        cls = lightgbm.LGBMRegressor if task == const.TASK_REGRESSION else lightgbm.LGBMClassifier
        return cls(n_estimators=50,
                   num_leaves=15,
                   max_depth=5,
                   subsample=0.5,
                   subsample_freq=1,
                   colsample_bytree=0.8,
                   reg_alpha=1,
                   reg_lambda=1,
                   importance_type='gain', )

    def default_dt():
        cls = DecisionTreeRegressor if task == const.TASK_REGRESSION else DecisionTreeClassifier
        return cls(min_samples_leaf=20, min_impurity_decrease=0.01)

    def default_rf():
        cls = RandomForestRegressor if task == const.TASK_REGRESSION else RandomForestClassifier
        return cls(min_samples_leaf=20, min_impurity_decrease=0.01)

    def default_dask_xgb():
        import dask_xgboost
        cls = dask_xgboost.XGBRegressor if task == const.TASK_REGRESSION else dask_xgboost.XGBClassifier
        return cls(max_depth=5,
                   n_estimators=50,
                   min_child_weight=3,
                   gamma=1,
                   # reg_alpha=0.1,
                   # reg_lambda=0.1,
                   subsample=0.6,
                   colsample_bytree=0.6,
                   eval_metric='auc',
                   # objective='binary:logitraw',
                   tree_method='approx', )

    if dex.is_dask_dataframe(X):
        try:
            estimator_ = default_dask_xgb()
            estimator_ = _wrap_predict_proba(estimator_)
        except ImportError:  # failed to import dask_xgboost
            estimator_ = default_gbm()
            estimator_ = dex.wrap_local_estimator(estimator_)
    else:
        if estimator is None or estimator == 'gbm':
            estimator_ = default_gbm()
        elif estimator == 'dt':
            estimator_ = default_dt()
        elif estimator == 'rf':
            estimator_ = default_rf()
        else:
            estimator_ = copy.deepcopy(estimator)

    return estimator_
