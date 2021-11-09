# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from sklearn.model_selection import train_test_split

from hypernets.utils import logging, const
from . import Experiment

logger = logging.get_logger(__name__)


class GeneralExperiment(Experiment):
    def __init__(self, hyper_model, X_train, y_train, X_eval=None, y_eval=None, X_test=None, eval_size=0.3,
                 task=None, id=None, callbacks=None, random_state=9527):
        super(GeneralExperiment, self).__init__(hyper_model, X_train, y_train, X_eval=X_eval,
                                                y_eval=y_eval, X_test=X_test, eval_size=eval_size, task=task,
                                                id=id, callbacks=callbacks, random_state=random_state)

    def train(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, **kwargs):
        """Run an experiment
        """
        self.step_start('data split')
        if X_eval is None or y_eval is None:
            stratify = y_train
            if self.task == const.TASK_REGRESSION:
                stratify = None
            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=self.eval_size,
                                                                random_state=self.random_state, stratify=stratify)
        self.step_end(output={'X_train.shape': X_train.shape,
                              'y_train.shape': y_train.shape,
                              'X_eval.shape': X_eval.shape,
                              'y_eval.shape': y_eval.shape,
                              'X_test.shape': None if X_test is None else X_test.shape})

        self.step_start('search')
        hyper_model.search(X_train, y_train, X_eval, y_eval, **kwargs)
        best_trial = hyper_model.get_best_trial()
        self.step_end(output={'best_trial': best_trial})

        self.step_start('load estimator')
        estimator = hyper_model.load_estimator(best_trial.model_file)
        self.step_end(output={'estimator': estimator})

        return estimator
