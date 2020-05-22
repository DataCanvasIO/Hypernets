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

    def _get_estimator(self, space):
        raise NotImplementedError

    def get_best_trail(self):
        return self.history.get_best()

    def _run_trial(self, space, trail_no, X, y, X_val, y_val, **kwargs):
        start_time = time.time()
        estimator = self._get_estimator(space)

        for callback in self.callbacks:
            callback.on_build_estimator(self, space, estimator, trail_no)
            callback.on_trail_begin(self, space, trail_no)

        estimator.fit(X, y, **kwargs)
        metrics = estimator.evaluate(X_val, y_val)
        reward = self.get_reward(metrics, self.reward_metric)
        self.searcher.update_result(space.space_id, space.get_assignable_param_values(), reward)
        elapsed = time.time() - start_time
        trail = Trail(space, trail_no, reward, elapsed)
        improved = self.history.append(trail)
        if improved:
            self.best_model = estimator.model

        for callback in self.callbacks:
            callback.on_trail_end(self, space, trail_no, reward, improved, elapsed)

        return estimator.model

    def get_reward(self, value, key=None):
        if key is None:
            key = 'reward'
        if isinstance(value, (float, int)):
            reward = value
        elif isinstance(value, dict) and key in value and isinstance(value[key], (float, int)):
            reward = value[key]
        else:
            raise ValueError(
                f'[value] should be (float/int) or a dict which has a key named "{key}" whose value is (float/int).')
        return reward

    def search(self, X, y, X_val, y_val, **kwargs):
        self.start_search_time = time.time()
        for trail_no in range(1, self.max_trails + 1):
            space = self.searcher.sample()
            try:
                self._run_trial(space, trail_no, X, y, X_val, y_val, **kwargs)
            except EarlyStoppingError:
                break
                # TODO: early stopping

    def final_train(self, space, X, y, **kwargs):
        pass
