# -*- coding:utf-8 -*-
"""

"""

from hypernets.core.search_space import ModuleSpace, Choice
from hypernets.core.ops import ConnectionSpace
from sklearn import impute, pipeline, compose, preprocessing as sk_pre

import numpy as np


class HyperEstimator(ModuleSpace):
    def __init__(self, task, fit_kwargs, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.task = task
        self.fit_kwargs = fit_kwargs
        self.estimator = None

    def _build_estimator(self, kwargs):
        raise NotImplementedError

    def _build(self):
        pv = self.param_values
        self.estimator = self._build_estimator(pv)

    def _compile(self, inputs):
        return self.estimator

    def _on_params_ready(self):
        pass


class LightGBMEstimator(HyperEstimator):
    def __init__(self, task, fit_kwargs, boosting_type='gbdt', num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=100,
                 subsample_for_bin=200000, objective=None, class_weight=None,
                 min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
                 subsample=1., subsample_freq=0, colsample_bytree=1.,
                 reg_alpha=0., reg_lambda=0., random_state=None,
                 n_jobs=-1, silent=True, importance_type='split', space=None, name=None, **kwargs):

        if boosting_type is not None and boosting_type != 'gbdt':
            kwargs['boosting_type'] = boosting_type
        if num_leaves is not None and num_leaves != 31:
            kwargs['num_leaves'] = num_leaves
        if max_depth is not None and max_depth != -1:
            kwargs['max_depth'] = max_depth
        if learning_rate is not None and learning_rate != 0.1:
            kwargs['learning_rate'] = learning_rate
        if n_estimators is not None and n_estimators != 100:
            kwargs['n_estimators'] = n_estimators
        if subsample_for_bin is not None and subsample_for_bin != 200000:
            kwargs['subsample_for_bin'] = subsample_for_bin
        if objective is not None:
            kwargs['objective'] = objective
        if class_weight is not None:
            kwargs['class_weight'] = class_weight
        if min_split_gain is not None and min_split_gain != 0.:
            kwargs['min_split_gain'] = min_split_gain
        if min_child_weight is not None and min_child_weight != 1e-3:
            kwargs['min_child_weight'] = min_child_weight
        if min_child_samples is not None and min_child_samples != 20:
            kwargs['min_child_samples'] = min_child_samples
        if subsample is not None and subsample != 1.:
            kwargs['subsample'] = subsample
        if subsample_freq is not None and subsample_freq != 0:
            kwargs['subsample_freq'] = subsample_freq
        if colsample_bytree is not None and colsample_bytree != 1.:
            kwargs['colsample_bytree'] = colsample_bytree
        if reg_alpha is not None and reg_alpha != 0.:
            kwargs['reg_alpha'] = reg_alpha
        if reg_lambda is not None and reg_lambda != 0.:
            kwargs['reg_lambda'] = reg_lambda
        if random_state is not None:
            kwargs['random_state'] = random_state
        if n_jobs is not None and n_jobs != -1:
            kwargs['n_jobs'] = n_jobs
        if silent is not None and silent != True:
            kwargs['silent'] = silent
        if importance_type is not None and importance_type != 'split':
            kwargs['importance_type'] = importance_type

        HyperEstimator.__init__(self, task, fit_kwargs, space, name, **kwargs)

    def _build_estimator(self, kwargs):
        import lightgbm
        if self.task == 'regression':
            lgbm = lightgbm.LGBMRegressor(**kwargs)
        else:
            lgbm = lightgbm.LGBMClassifier(**kwargs)
        return lgbm
