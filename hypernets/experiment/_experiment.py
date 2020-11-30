# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import time


class Experiment():
    def __init__(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3,
                 random_state=9527):
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
        self.random_state = random_state

        self.start_time = None
        self.end_time = None

    def run(self, **kwargs):
        self.start_time = time.time()

        model = self.train(self.hyper_model, self.X_train, self.y_train, self.X_test, X_eval=self.X_eval,
                           y_eval=self.y_eval, **kwargs)
        self.end_time = time.time()
        return model

    def data_split(self, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3):
        raise NotImplementedError

    def train(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, **kwargs):
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
        """
        raise NotImplementedError

    def export_model(self):
        raise NotImplementedError

    @property
    def elapsed(self):
        if self.start_time is None or self.end_time is None:
            return -1
        else:
            return self.end_time - self.start_time
