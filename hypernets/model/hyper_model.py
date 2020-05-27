# -*- coding:utf-8 -*-
"""

"""
import time
from ..core.callbacks import EarlyStoppingError
from ..core.trial import *


class HyperModel():
    def __init__(self, searcher, dispatcher=None, callbacks=[], max_trails=10, reward_metric=None):
        # self.searcher = self._build_searcher(searcher, space_fn)
        self.searcher = searcher
        self.dispatcher = dispatcher
        self.callbacks = callbacks
        self.max_trails = max_trails
        self.reward_metric = reward_metric
        self.history = TrailHistory(searcher.optimize_direction)
        self.best_model = None
        self.start_search_time = None

    def sample_space(self):
        return self.searcher.sample()

    def _get_estimator(self, space_sample):
        raise NotImplementedError

    def _run_trial(self, space_sample, trail_no, X, y, X_val, y_val, **fit_kwargs):
        start_time = time.time()
        estimator = self._get_estimator(space_sample)

        for callback in self.callbacks:
            callback.on_build_estimator(self, space_sample, estimator, trail_no)
            callback.on_trail_begin(self, space_sample, trail_no)

        estimator.fit(X, y, **fit_kwargs)
        metrics = estimator.evaluate(X_val, y_val)
        reward = self._get_reward(metrics, self.reward_metric)
        self.searcher.update_result(space_sample, reward)
        elapsed = time.time() - start_time
        trail = Trail(space_sample, trail_no, reward, elapsed)
        improved = self.history.append(trail)
        if improved:
            self.best_model = estimator.model

        for callback in self.callbacks:
            callback.on_trail_end(self, space_sample, trail_no, reward, improved, elapsed)

        return estimator.model

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
        elif isinstance(value, dict) and key in value and cast_float(value[key]) is not None:
            reward = cast_float(value[key])
        else:
            raise ValueError(
                f'[value] should be a numeric or a dict which has a key named "{key}" whose value is a numeric.')
        return reward

    def get_best_trail(self):
        return self.history.get_best()

    def search(self, X, y, X_val, y_val, **fit_kwargs):
        self.start_search_time = time.time()
        for trail_no in range(1, self.max_trails + 1):
            space_sample = self.searcher.sample()
            try:
                self._run_trial(space_sample, trail_no, X, y, X_val, y_val, **fit_kwargs)
            except EarlyStoppingError:
                break
                # TODO: early stopping

    def final_train(self, space_sample, X, y, **kwargs):
        estimator = self._get_estimator(space_sample)
        estimator.fit(X, y, **kwargs)
        return estimator
