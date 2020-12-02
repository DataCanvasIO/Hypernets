# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from sklearn.model_selection import train_test_split

from hypernets.utils import logging
from . import Experiment

logger = logging.get_logger(__name__)


class GeneralExperiment(Experiment):
    def __init__(self, task, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3,
                 callbacks=None, random_state=9527):
        super(GeneralExperiment, self).__init__(task, hyper_model, X_train, y_train, X_test, X_eval=X_eval,
                                                y_eval=y_eval,
                                                eval_size=eval_size,
                                                callbacks=callbacks, random_state=random_state)

    def train(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, eval_size=0.3, **kwargs):
        """Run an experiment

        Arguments
        ---------
        max_trails :

        """
        self.step_start('data split')
        if X_eval or y_eval is None:
            stratify = y_train
            if self.task == 'regression':
                stratify = None
            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=eval_size,
                                                                random_state=self.random_state, stratify=stratify)
        self.step_end()

        self.step_start('search')
        hyper_model.search(X_train, y_train, X_eval, y_eval, **kwargs)
        best_trial = hyper_model.get_best_trail()
        self.step_end(output={'best_trail': best_trial})

        self.step_start('load estimator')
        self.estimator = hyper_model.load_estimator(best_trial.model_file)
        self.step_end(output={'estimator': self.estimator})

        return self.estimator
