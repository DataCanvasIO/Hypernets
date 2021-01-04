# -*- coding:utf-8 -*-
"""

"""
import hashlib
import time
import traceback
from collections import UserDict

from ..core.meta_learner import MetaLearner
from ..core.trial import *
from ..dispatchers import get_dispatcher
from ..utils import logging

logger = logging.get_logger(__name__)


class HyperModel():
    def __init__(self, searcher, dispatcher=None, callbacks=[], reward_metric=None, task=None):
        # self.searcher = self._build_searcher(searcher, space_fn)
        self.searcher = searcher
        self.dispatcher = dispatcher
        self.callbacks = callbacks
        self.reward_metric = reward_metric
        self.history = TrailHistory(searcher.optimize_direction)
        self.start_search_time = None
        self.task = task

    def _get_estimator(self, space_sample):
        raise NotImplementedError

    def load_estimator(self, model_file):
        raise NotImplementedError

    def _run_trial(self, space_sample, trail_no, X, y, X_eval, y_eval, model_file, **fit_kwargs):

        start_time = time.time()
        estimator = self._get_estimator(space_sample)

        for callback in self.callbacks:
            callback.on_build_estimator(self, space_sample, estimator, trail_no)
        #     callback.on_trail_begin(self, space_sample, trail_no)
        fit_succeed = False
        try:
            estimator.fit(X, y, **fit_kwargs)
            fit_succeed = True
        except Exception as e:
            logger.error('Estimator fit failed!')
            logger.error(e)
            track = traceback.format_exc()
            logger.error(track)

        if fit_succeed:
            metrics = estimator.evaluate(X_eval, y_eval, metrics=[self.reward_metric], **fit_kwargs)
            reward = self._get_reward(metrics, self.reward_metric)

            if model_file is None or len(model_file) == 0:
                model_file = '%05d_%s.pkl' % (trail_no, space_sample.space_id)
            estimator.save(model_file)

            elapsed = time.time() - start_time

            trail = Trail(space_sample, trail_no, reward, elapsed, model_file)

            # improved = self.history.append(trail)

            self.searcher.update_result(space_sample, reward)

            # for callback in self.callbacks:
            #     callback.on_trail_end(self, space_sample, trail_no, reward, improved, elapsed)
        else:
            # for callback in self.callbacks:
            #     callback.on_trail_error(self, space_sample, trail_no)

            elapsed = time.time() - start_time
            trail = Trail(space_sample, trail_no, 0, elapsed)

        return trail

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

    def get_best_trail(self):
        return self.history.get_best()

    @property
    def best_reward(self):
        best = self.get_best_trail()
        if best is not None:
            return best.reward
        else:
            return None

    @property
    def best_trail_no(self):
        best = self.get_best_trail()
        if best is not None:
            return best.trail_no
        else:
            return None

    def get_top_trails(self, top_n):
        return self.history.get_top(top_n)

    def _before_search(self):
        pass

    def _after_search(self, last_trail_no):
        pass

    def search(self, X, y, X_eval, y_eval, max_trails=10, dataset_id=None, trail_store=None, **fit_kwargs):
        self.start_search_time = time.time()

        self.task, _ = self.infer_task_type(y)

        if dataset_id is None:
            dataset_id = self.generate_dataset_id(X, y)
        if self.searcher.use_meta_learner:
            self.searcher.set_meta_learner(MetaLearner(self.history, dataset_id, trail_store))

        self._before_search()

        dispatcher = self.dispatcher if self.dispatcher else get_dispatcher(self)
        trail_no = dispatcher.dispatch(self, X, y, X_eval, y_eval, max_trails, dataset_id, trail_store, **fit_kwargs)

        self._after_search(trail_no)

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
        for trail in trials:
            configurations.append(self.export_trail_configuration(trail))
        return configurations

    def export_trail_configuration(self, trail):
        raise NotImplementedError

    def infer_task_type(self, y):
        if len(y.shape) > 1 and y.shape[-1] > 1:
            labels = list(range(y.shape[-1]))
            task = 'multilable'
            return task, labels

        uniques = set(y)
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
