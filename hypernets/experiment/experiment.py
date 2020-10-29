# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import time
from sklearn.model_selection import train_test_split


class Experiment():
    def __init__(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3, strategy=None):
        self.id = None
        self.title = None
        self.description = None
        self.dataset_id = None
        self.path = None

        self.hyper_model = hyper_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.eval_size = eval_size

        self.start_time = None
        self.end_time = None

        self.stragety = strategy if strategy is not None else ExperimentStragety()

    def run(self, **kwargs):
        self.start_time = time.time()
        X_train, y_train, X_test, X_eval, y_eval = self.stragety.data_split(self.X_train, self.y_train,
                                                                            self.X_test, self.X_eval,
                                                                            self.y_eval, self.eval_size)
        model = self.stragety.train(self.hyper_model, X_train, y_train, X_test, X_eval=X_eval, y_eval=y_eval, **kwargs)
        self.end_time = time.time()
        return model

    @property
    def elapsed(self):
        if self.start_time is None or self.end_time is None:
            return -1
        else:
            return self.end_time - self.start_time


class ExperimentStragety():
    def data_split(self, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3):
        raise NotImplementedError

    def train(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, **kwargs):
        raise NotImplementedError


class DefaultExperimentStragety(ExperimentStragety):
    def __init__(self, random_state=9527):
        self.random_state = random_state

    def data_split(self, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3):
        if X_eval or y_eval is None:
            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=eval_size,
                                                                random_state=self.random_state, stratify=y_train)
        return X_train, y_train, X_test, X_eval, y_eval

    def train(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3, **kwargs):
        """Run an experiment

        Arguments
        ---------
        hyper_model : HyperModel
        X_train :
        y_train :
        X_test :
        X_eval :
        y_eval :
        eval_size :

        max_trails :

        """
        if X_eval or y_eval is None:
            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=eval_size,
                                                                random_state=self.random_state, stratify=y_train)
        max_trails = kwargs.get('max_trails')
        if max_trails is None:
            max_trails = 10

        hyper_model.search(X_train, y_train, X_eval, y_eval, max_trails=max_trails)
        best_trial = hyper_model.get_best_trail()

        estimator = hyper_model.final_train(best_trial.space_sample, X_train, y_train)
        return estimator
