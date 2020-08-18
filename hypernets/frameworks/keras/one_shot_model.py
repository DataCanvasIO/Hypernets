# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from .hyper_keras import HyperKeras
from ...core.trial import Trail
import time


class OneShotModel(HyperKeras):
    def __init__(self, searcher, optimizer, loss, metrics, epochs=320, batch_size=64, controller_train_steps=50,
                 dispatcher=None, callbacks=[], reward_metric=None, max_model_size=0, visualization=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.controller_train_steps = controller_train_steps
        super(OneShotModel, self).__init__(searcher, optimizer, loss, metrics,
                                           dispatcher=dispatcher,
                                           callbacks=callbacks,
                                           reward_metric=reward_metric,
                                           max_model_size=max_model_size,
                                           one_shot_mode=True,
                                           one_shot_training_sampler=searcher,
                                           visualization=visualization)

    def search(self, X, y, X_val, y_val, max_trails=10, dataset_id=None, trail_store=None, **fit_kwargs):
        self.start_search_time = time.time()

        trail_no = 1
        for epoch in range(self.epochs):
            if trail_no >= max_trails:
                break

            print(f'One-shot model epoch({epoch}) training...')
            self.fit_one_shot_model_epoch(X, y, batch_size=self.batch_size, epoch=epoch)

            print(f'Controller epoch({epoch}) training...')
            for controller_step in range(self.controller_train_steps):
                start_time = time.time()
                space_sample = self.searcher.sample()
                estimator = self._get_estimator(space_sample)
                for callback in self.callbacks:
                    callback.on_build_estimator(self, space_sample, estimator, trail_no)
                    callback.on_trail_begin(self, space_sample, trail_no)

                metrics = estimator.evaluate(X_val, y_val, batch_size=self.batch_size, metrics=[self.reward_metric])
                reward = self._get_reward(metrics, self.reward_metric)
                elapsed = time.time() - start_time
                trail = Trail(space_sample, trail_no, reward, elapsed)
                improved = self.history.append(trail)

                if improved:
                    self.best_model = estimator.model
                self.searcher.update_result(space_sample, reward)

                for callback in self.callbacks:
                    callback.on_trail_end(self, space_sample, trail_no, reward, improved, elapsed)
                trail_no += 1

    def derive_arch(self, X_test, y_test, num=10):
        space_sample = self.searcher.sample()
        estimator = self._get_estimator()
        metrics = estimator.evaluate(X_test, y_test, batch_size=self.batch_size, metrics=[self.reward_metric])
        reward = self._get_reward(metrics, self.reward_metric)
        print('>' * 50)
        estimator.summary()
        print(f'Reward on test set: {reward}')
        print('>' * 50)
