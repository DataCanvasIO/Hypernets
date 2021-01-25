# -*- coding:utf-8 -*-
"""

"""
import hashlib
import time
import traceback
from collections import UserDict
import numpy as np

from ..core.meta_learner import MetaLearner
from ..core.trial import *
from ..dispatchers import get_dispatcher
from ..utils import logging

logger = logging.get_logger(__name__)


class HyperModel():
    def __init__(self, searcher, dispatcher=None, callbacks=None, reward_metric=None, task=None):
        """

        :param searcher:
        :param dispatcher:
        :param callbacks:
        :param reward_metric:
        :param task:
        """
        # self.searcher = self._build_searcher(searcher, space_fn)
        self.searcher = searcher
        self.dispatcher = dispatcher
        self.callbacks = callbacks if callbacks is not None else []
        self.reward_metric = reward_metric
        self.history = TrialHistory(searcher.optimize_direction)
        self.start_search_time = None
        self.task = task

    def _get_estimator(self, space_sample):
        raise NotImplementedError

    def load_estimator(self, model_file):
        raise NotImplementedError

    def _run_trial(self, space_sample, trial_no, X, y, X_eval, y_eval, cv=False, num_folds=3, model_file=None,
                   **fit_kwargs):

        start_time = time.time()
        estimator = self._get_estimator(space_sample)

        for callback in self.callbacks:
            callback.on_build_estimator(self, space_sample, estimator, trial_no)
        #     callback.on_trial_begin(self, space_sample, trial_no)
        fit_succeed = False
        scores = None
        oof = None
        try:
            if cv:
                scores, oof = estimator.fit_cross_validation(X, y, stratified=True, num_folds=num_folds,
                                                             shuffle=False, random_state=9527,
                                                             metrics=[self.reward_metric],
                                                             **fit_kwargs)
            else:
                estimator.fit(X, y, **fit_kwargs)
            fit_succeed = True
        except Exception as e:
            logger.error('Estimator fit failed!')
            logger.error(e)
            track = traceback.format_exc()
            logger.error(track)

        if fit_succeed:
            if scores is None:
                scores = estimator.evaluate(X_eval, y_eval, metrics=[self.reward_metric], **fit_kwargs)
            reward = self._get_reward(scores, self.reward_metric)

            if model_file is None or len(model_file) == 0:
                model_file = '%05d_%s.pkl' % (trial_no, space_sample.space_id)
            estimator.save(model_file)

            elapsed = time.time() - start_time

            trial = Trial(space_sample, trial_no, reward, elapsed, model_file)
            if oof is not None:
                trial.memo['oof'] = oof

            # improved = self.history.append(trial)

            self.searcher.update_result(space_sample, reward)

            # for callback in self.callbacks:
            #     callback.on_trial_end(self, space_sample, trial_no, reward, improved, elapsed)
        else:
            # for callback in self.callbacks:
            #     callback.on_trial_error(self, space_sample, trial_no)

            elapsed = time.time() - start_time
            trial = Trial(space_sample, trial_no, 0, elapsed)

        return trial

    def _get_reward(self, value, key=None):
        def cast_float(value):
            try:
                fv = float(value)
                return fv
            except TypeError:
                return None

        if key is None:
            key = 'reward'

        fv = cast_float(value)
        if fv is not None:
            reward = fv
        elif (isinstance(value, dict) or isinstance(value, UserDict)) and key in value and cast_float(
                value[key]) is not None:
            reward = cast_float(value[key])
        else:
            raise ValueError(
                f'[value] should be a numeric or a dict which has a key named "{key}" whose value is a numeric.')
        return reward

    def get_best_trial(self):
        return self.history.get_best()

    @property
    def best_reward(self):
        best = self.get_best_trial()
        if best is not None:
            return best.reward
        else:
            return None

    @property
    def best_trial_no(self):
        best = self.get_best_trial()
        if best is not None:
            return best.trial_no
        else:
            return None

    def get_top_trials(self, top_n):
        return self.history.get_top(top_n)

    def _before_search(self):
        pass

    def _after_search(self, last_trial_no):
        pass

    def search(self, X, y, X_eval, y_eval, cv=False, num_folds=3, max_trials=10, dataset_id=None, trial_store=None,
               **fit_kwargs):
        """
        :param X: Pandas or Dask DataFrame, feature data for training
        :param y: Pandas or Dask Series, target values for training
        :param X_eval: (Pandas or Dask DataFrame) or None, feature data for evaluation
        :param y_eval: (Pandas or Dask Series) or None, target values for evaluation
        :param cv: Optional, int(default=False), If set to `true`, use cross-validation instead of evaluation set reward to guide the search process
        :param num_folds: Optional, int(default=3), Number of cross-validated folds, only valid when cv is true
        :param max_trials: Optional, int(default=10), The upper limit of the number of search trials, the search process stops when the number is exceeded
        :param dataset_id:
        :param trial_store:
        :param fit_kwargs: Optional, dict, parameters for fit method of model
        :return:
        """
        self.start_search_time = time.time()

        self.task, _ = self.infer_task_type(y)

        if dataset_id is None:
            dataset_id = self.generate_dataset_id(X, y)
        if self.searcher.use_meta_learner:
            self.searcher.set_meta_learner(MetaLearner(self.history, dataset_id, trial_store))

        self._before_search()

        dispatcher = self.dispatcher if self.dispatcher else get_dispatcher(self)
        trial_no = dispatcher.dispatch(self, X, y, X_eval, y_eval, cv, num_folds, max_trials, dataset_id, trial_store,
                                       **fit_kwargs)

        self._after_search(trial_no)

    def generate_dataset_id(self, X, y):
        repr = ''
        if X is not None:
            if isinstance(X, list):
                repr += f'X len({len(X)})|'
            if hasattr(X, 'shape'):
                repr += f'X shape{X.shape}|'
            if hasattr(X, 'dtypes'):
                repr += f'x.dtypes({list(X.dtypes)})|'

        if y is not None:
            if isinstance(y, list):
                repr += f'y len({len(y)})|'
            if hasattr(y, 'shape'):
                repr += f'y shape{y.shape}|'

            if hasattr(y, 'dtype'):
                repr += f'y.dtype({y.dtype})|'

        sign = hashlib.md5(repr.encode('utf-8')).hexdigest()
        return sign

    def final_train(self, space_sample, X, y, **kwargs):
        estimator = self._get_estimator(space_sample)
        estimator.fit(X, y, **kwargs)
        return estimator

    def export_configuration(self, trials):
        configurations = []
        for trial in trials:
            configurations.append(self.export_trial_configuration(trial))
        return configurations

    def export_trial_configuration(self, trial):
        raise NotImplementedError

    def infer_task_type(self, y):
        if len(y.shape) > 1 and y.shape[-1] > 1:
            labels = list(range(y.shape[-1]))
            task = 'multilable'
            return task, labels

        if hasattr(y, 'unique'):
            uniques = set(y.unique())
        else:
            uniques = set(y)

        if uniques.__contains__(np.nan):
            uniques.remove(np.nan)
        n_unique = len(uniques)
        labels = []

        if n_unique == 2:
            logger.info(f'2 class detected, {uniques}, so inferred as a [binary classification] task')
            task = 'binary'  # TASK_BINARY
            labels = sorted(uniques)
        else:
            if y.dtype == 'float':
                logger.info(f'Target column type is float, so inferred as a [regression] task.')
                task = 'regression'
            else:
                if n_unique > 1000:
                    if 'int' in y.dtype:
                        logger.info(
                            'The number of classes exceeds 1000 and column type is int, so inferred as a [regression] task ')
                        task = 'regression'
                    else:
                        raise ValueError(
                            'The number of classes exceeds 1000, please confirm whether your predict target is correct ')
                else:
                    logger.info(f'{n_unique} class detected, inferred as a [multiclass classification] task')
                    task = 'multiclass'
                    labels = sorted(uniques)
        return task, labels
