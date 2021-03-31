# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from . import BaseEstimator
from sklearn.metrics import get_scorer

from hypergbm import HyperGBM, CompeteExperiment
from hypergbm.search_space import search_space_general, search_space_feature_gen
from hypernets.core.searcher import OptimizeDirection
from hypernets.experiment import GeneralExperiment, ConsoleCallback
from hypernets.searchers import RandomSearcher, EvolutionSearcher, MCTSSearcher
from hypernets.core import EarlyStoppingCallback, SummaryCallback


class HyperGBMEstimator(BaseEstimator):
    def __init__(self, task, scorer, reward_metric='auc', optimize_direction='max', mode='one-stage', max_trials=30, use_cache=True, earlystop_rounds=30,
                 time_limit=3600, expected_reward=None,
                 drop_feature_with_collinearity=False,
                 search_space_fn=None, ensemble_size=10, use_meta_learner=False, eval_size=0.3,
                 train_test_split_strategy=None,
                 cv=True, num_folds=3, class_balancing=None,
                 retrain_on_wholedata=False,
                 two_stage_importance_selection=True,
                 pseudo_labeling=False,
                 pseudo_labeling_proba_threshold=0.8,
                 pseudo_labeling_resplit=False,
                 **kwargs):
        super(HyperGBMEstimator, self).__init__(task)
        if kwargs.get('name') is not None:
            self.name = kwargs['name']
        else:
            self.name = 'HyperGBM'
        self.scorer = scorer
        self.reward_metric = reward_metric
        self.optimize_direction = optimize_direction
        self.mode = mode
        self.kwargs = kwargs
        self.estimator = None
        self.max_trials = max_trials
        self.use_cache = use_cache
        self.earlystop_rounds = earlystop_rounds
        self.time_limit = time_limit
        self.expected_reward = expected_reward
        self.cv = cv
        self.num_folds = num_folds
        self.search_space_fn = search_space_fn if search_space_fn is not None else lambda: search_space_general(
            early_stopping_rounds=10, verbose=0, class_balancing=class_balancing)
        self.ensemble_size = ensemble_size
        self.experiment = None
        self.use_meta_learner = use_meta_learner
        self.eval_size = eval_size
        self.train_test_split_strategy = train_test_split_strategy
        self.two_stage_importance_selection = two_stage_importance_selection
        self.retrain_on_wholedata = retrain_on_wholedata
        self.pseudo_labeling = pseudo_labeling
        self.pseudo_labeling_proba_threshold = pseudo_labeling_proba_threshold
        self.pseudo_labeling_resplit = pseudo_labeling_resplit
        self.drop_feature_with_collinearity = drop_feature_with_collinearity

    def train(self, X, y, X_test):
        # searcher = MCTSSearcher(self.search_space_fn, use_meta_learner=self.use_meta_learner, max_node_space=10,
        #                         candidates_size=10,
        #                         optimize_direction=OptimizeDirection.Maximize)
        searcher = EvolutionSearcher(self.search_space_fn,
                                     optimize_direction=self.optimize_direction, population_size=30, sample_size=10,
                                     regularized=True, candidates_size=10, use_meta_learner=self.use_meta_learner)
        # searcher = RandomSearcher(lambda: search_space_general(early_stopping_rounds=20, verbose=0),
        #                     optimize_direction=OptimizeDirection.Maximize)
        es = EarlyStoppingCallback(self.earlystop_rounds, self.optimize_direction, time_limit=self.time_limit,
                                   expected_reward=self.expected_reward)

        hk = HyperGBM(searcher, reward_metric=self.reward_metric, cache_dir=f'hypergbm_cache', clear_cache=True,
                      callbacks=[es, SummaryCallback()])

        log_callback = ConsoleCallback()
        self.experiment = CompeteExperiment(hk, X, y, X_test=X_test, eval_size=self.eval_size,
                                            train_test_split_strategy=self.train_test_split_strategy,
                                            cv=self.cv, num_folds=self.num_folds,
                                            callbacks=[],
                                            scorer=get_scorer(self.scorer),
                                            drop_feature_with_collinearity=self.drop_feature_with_collinearity,
                                            drift_detection=True,
                                            n_est_feature_importance=5,
                                            importance_threshold=1e-5,
                                            two_stage_importance_selection=self.two_stage_importance_selection,
                                            ensemble_size=self.ensemble_size,
                                            pseudo_labeling=self.pseudo_labeling,
                                            pseudo_labeling_proba_threshold=self.pseudo_labeling_proba_threshold,
                                            pseudo_labeling_resplit=self.pseudo_labeling_resplit,
                                            retrain_on_wholedata=self.retrain_on_wholedata,
                                            )
        self.estimator = self.experiment.run(use_cache=self.use_cache, max_trials=self.max_trials)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return self.estimator.predict(X)
