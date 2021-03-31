# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from .hypergbm import HyperGBMEstimator
from .auto_sklearn import AutoSklearnEstimator
from .h2o import H2OEstimator
from .hyperdt import HyperDTEstimator
from . import Evaluator
from ..datasets import dsutils
from hypergbm.search_space import search_space_feature_gen
import pandas as pd


class Test_Evaluator():
    def test_hypergbm_multiclass(self):
        def get_data(ds_name, task):
            dir = '/Users/jack/workspace/aps/notebook/hypergbm'
            train = pd.read_csv(f'{dir}/datasets/cooka_data/{task}/{ds_name}/train.csv')
            test = pd.read_csv(f'{dir}/datasets/cooka_data/{task}/{ds_name}/test.csv')
            return train, test

        task = 'multiclass'
        target = 'Class'
        X = get_data('yeast', task)
        # X = dsutils.load_glass_uci()
        hypergbm_estimator = HyperGBMEstimator(task=task, scorer='roc_auc_ovo',
                                               eval_size=0.1,
                                               cv=False,
                                               max_trials=3)
        autosklearn_estimator = AutoSklearnEstimator(task=task, time_left_for_this_task=30,
                                                     per_run_time_limit=10)
        h2o_estimator = H2OEstimator(task=task)
        hyperdt_estimator = HyperDTEstimator(task=task, reward_metric='AUC', max_trials=3, epochs=1)
        evaluator = Evaluator()
        result = evaluator.evaluate(X,
                                    target=target,
                                    task='multiclas',
                                    estimators=[
                                        # autosklearn_estimator,
                                        hypergbm_estimator,
                                        # h2o_estimator,
                                        # hyperdt_estimator,
                                    ],

                                    scorers=['roc_auc_ovo'],
                                    test_size=0.3,
                                    random_state=9527)
        assert result

    def test_splitdata(self):
        X = dsutils.load_blood()  # .load_bank().head(1000)
        task = 'binary'
        hypergbm_estimator = HyperGBMEstimator(task=task, scorer='roc_auc_ovo')

        evaluator = Evaluator()
        result = evaluator.evaluate((X[:-100], X[-100:]),
                                    target='Class',
                                    task=task,
                                    estimators=[
                                        # autosklearn_estimator,
                                        hypergbm_estimator,
                                        # h2o_estimator,
                                        # hyperdt_estimator,
                                        # hypergbm_estimator_fg
                                    ],
                                    scorers=['accuracy', 'roc_auc_ovo'],
                                    test_size=0.3,
                                    random_state=9527)

        assert result

    def test_hypergbm_cv_binary(self):
        X = dsutils.load_telescope()  # .load_bank().head(1000)
        task = 'binary'
        hypergbm_estimator = HyperGBMEstimator(task=task, scorer='roc_auc_ovo', cv=True, num_folds=3)
        hypergbm_estimator = HyperGBMEstimator(task=task, scorer='roc_auc_ovo', mode='two-stage', max_trials=300,
                                               earlystop_rounds=10,
                                               use_cache=False,
                                               drop_feature_with_collinearity=False,
                                               ensemble_size=20, use_meta_learner=False, eval_size=0.1,
                                               retrain_on_wholedata=False,
                                               # class_balancing='sample_weight',
                                               cv=True, num_folds=3,
                                               two_stage_importance_selection=False,
                                               pseudo_labeling=False, pseudo_labeling_proba_threshold=0.6,
                                               pseudo_labeling_resplit=False,
                                               )
        evaluator = Evaluator()
        result = evaluator.evaluate(X,
                                    target='Class',
                                    task=task,
                                    estimators=[
                                        hypergbm_estimator
                                    ],
                                    scorers=['roc_auc_ovo'],
                                    test_size=0.3,
                                    random_state=9527)
        assert result

    def test_all_binary(self):
        X = dsutils.load_blood()  # .load_bank().head(1000)
        task = 'binary'
        hypergbm_estimator = HyperGBMEstimator(task=task, scorer='roc_auc_ovo')
        hypergbm_estimator_fg = HyperGBMEstimator(task=task, scorer='roc_auc_ovo', max_trials=3,
                                                  search_space_fn=lambda: search_space_feature_gen(
                                                      early_stopping_rounds=20, verbose=0, task=task))

        autosklearn_estimator = AutoSklearnEstimator(task=task, time_left_for_this_task=30,
                                                     per_run_time_limit=10)
        h2o_estimator = H2OEstimator(task=task)
        hyperdt_estimator = HyperDTEstimator(task=task, reward_metric='AUC', max_trials=3, epochs=1)
        evaluator = Evaluator()
        result = evaluator.evaluate(X,
                                    target='Class',
                                    task=task,
                                    estimators=[
                                        # autosklearn_estimator,
                                        # hypergbm_estimator,
                                        # h2o_estimator,
                                        hyperdt_estimator,
                                        # hypergbm_estimator_fg
                                    ],
                                    scorers=['accuracy', 'roc_auc_ovo'],
                                    test_size=0.3,
                                    random_state=9527)
        assert result
