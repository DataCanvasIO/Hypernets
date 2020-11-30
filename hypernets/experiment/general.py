# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from sklearn.model_selection import train_test_split

from . import Experiment


class GeneralExperiment(Experiment):
    def __init__(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3,
                 random_state=9527):
        super().__init__(hyper_model, X_train, y_train, X_test, X_eval=X_eval, y_eval=y_eval, eval_size=eval_size,
                         random_state=random_state)

    def train(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3, **kwargs):
        """Run an experiment

        Arguments
        ---------
        max_trails :

        """
        if X_eval or y_eval is None:
            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=eval_size,
                                                                random_state=self.random_state, stratify=y_train)

        hyper_model.search(X_train, y_train, X_eval, y_eval, **kwargs)
        best_trial = hyper_model.get_best_trail()

        self.estimator = hyper_model.final_train(best_trial.space_sample, X_train, y_train)
        return self.estimator
