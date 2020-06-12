# -*- coding:utf-8 -*-
"""

"""
import time
from ..core.callbacks import EarlyStoppingError
from ..core.trial import *


class HyperModel():
    def __init__(self, searcher, dispatcher=None, callbacks=[], max_trails=10, reward_metric=None, trail_store=None):
        # self.searcher = self._build_searcher(searcher, space_fn)
        self.searcher = searcher
        self.dispatcher = dispatcher
        self.callbacks = callbacks
        self.max_trails = max_trails
        self.reward_metric = reward_metric
        self.history = TrailHistory(searcher.optimize_direction)
        self.best_model = None
        self.start_search_time = None
        self.trail_store = trail_store

    def sample_space(self):
        return self.searcher.sample(self.history)

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
        elapsed = time.time() - start_time
        trail = Trail(space_sample, trail_no, reward, elapsed)
        improved = self.history.append(trail)
        if improved:
            self.best_model = estimator.model

        self.searcher.update_result(space_sample, reward)

        for callback in self.callbacks:
            callback.on_trail_end(self, space_sample, trail_no, reward, improved, elapsed)

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
        elif isinstance(value, dict) and key in value and cast_float(value[key]) is not None:
            reward = cast_float(value[key])
        else:
            raise ValueError(
                f'[value] should be a numeric or a dict which has a key named "{key}" whose value is a numeric.')
        return reward

    def get_best_trail(self):
        return self.history.get_best()

    def search(self, X, y, X_val, y_val, dataset_id=None, **fit_kwargs):
        if dataset_id is None:
            dataset_id = self.generate_dataset_id(X, y)
        self.start_search_time = time.time()
        for trail_no in range(1, self.max_trails + 1):
            space_sample = self.sample_space()
            try:
                if self.trail_store is not None:
                    trail = self.trail_store.get(dataset_id, space_sample)
                    if trail is not None:
                        reward = trail.reward
                        elapsed = trail.elapsed
                        trail = Trail(space_sample, trail_no, reward, elapsed)
                        improved = self.history.append(trail)
                        if improved:
                            self.best_model = None
                        self.searcher.update_result(space_sample, reward)
                        for callback in self.callbacks:
                            callback.on_skip_trail(self, space_sample, trail_no, 'hit_history', reward, improved,
                                                   elapsed)
                            continue
                trail = self._run_trial(space_sample, trail_no, X, y, X_val, y_val, **fit_kwargs)
                print(f'----------------------------------------------------------------')
                print(f'space signatures: {self.history.get_space_signatures()}')
                print(f'----------------------------------------------------------------')
                if self.trail_store is not None:
                    self.trail_store.put(dataset_id, trail)
            except EarlyStoppingError:
                break
                # TODO: early stopping

    def generate_dataset_id(self, X: object, y):
        return str(X.shape)

    def final_train(self, space_sample, X, y, **kwargs):
        estimator = self._get_estimator(space_sample)
        estimator.fit(X, y, **kwargs)
        return estimator
