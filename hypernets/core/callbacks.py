# -*- coding:utf-8 -*-
"""

"""
import datetime
import json
import os
import time

import numpy as np

from ..utils import logging, fs

logger = logging.get_logger(__name__)


class Callback():
    def __init__(self):
        pass

    def on_build_estimator(self, hyper_model, space, estimator, trial_no):
        pass

    def on_trial_begin(self, hyper_model, space, trial_no):
        pass

    def on_trial_end(self, hyper_model, space, trial_no, reward, improved, elapsed):
        pass

    def on_trial_error(self, hyper_model, space, trial_no):
        pass

    def on_skip_trial(self, hyper_model, space, trial_no, reason, reward, improved, elapsed):
        pass


class EarlyStoppingError(RuntimeError):
    def __init__(self, *arg):
        self.args = arg


class EarlyStoppingCallback(Callback):
    def __init__(self, max_no_improvement_trials=0, mode='min', min_delta=0, time_limit=None, expected_reward=None):
        super(Callback, self).__init__()
        # assert time_limit is None or time_limit > 60, 'If `time_limit` is not None, it must be greater than 60.'

        self.max_no_improvement_trials = max_no_improvement_trials
        self.mode = mode
        self.min_delta = min_delta
        self.best_reward = None
        self.best_trial_no = None
        self.counter_no_improvement_trials = 0
        self.time_limit = time_limit
        self.expected_reward = expected_reward
        self.start_time = None
        if mode == 'min':
            self.op = np.less
        elif mode == 'max':
            self.op = np.greater
        else:
            raise ValueError(f'Unsupported mode:{mode}')

    def on_trial_begin(self, hyper_model, space, trial_no):
        if self.start_time is None:
            self.start_time = time.time()

    def on_trial_end(self, hyper_model, space, trial_no, reward, improved, elapsed):
        if self.time_limit is not None and self.time_limit > 0:
            time_total = time.time() - self.start_time
            if time_total > self.time_limit:
                msg = 'The time limit has been exceeded, stop early.\r\n'
                msg += f'Early stopping on trial : {trial_no}, best reward: {self.best_reward}, best_trial: {self.best_trial_no}'
                if logger.is_info_enabled():
                    logger.info(msg)
                raise EarlyStoppingError(msg)

        if self.expected_reward is not None and self.expected_reward != 0.0:
            if self.op(reward, self.expected_reward):
                msg = 'Has met the expected reward, stop early.\r\n'
                msg += f'Early stopping on trial : {trial_no}, best reward: {self.best_reward}, best_trial: {self.best_trial_no}'
                if logger.is_info_enabled():
                    logger.info(msg)
                raise EarlyStoppingError(msg)

        if self.max_no_improvement_trials is not None and self.max_no_improvement_trials > 0:
            if self.best_reward is None:
                self.best_reward = reward
                self.best_trial_no = trial_no
            else:
                if self.op(reward, self.best_reward - self.min_delta):
                    self.best_reward = reward
                    self.best_trial_no = trial_no
                    self.counter_no_improvement_trials = 0
                else:
                    self.counter_no_improvement_trials += 1
                    if self.counter_no_improvement_trials >= self.max_no_improvement_trials:
                        msg = f'Early stopping on trial : {trial_no}, best reward: {self.best_reward}, best_trial: {self.best_trial_no}'
                        if logger.is_info_enabled():
                            logger.info(msg)
                        raise EarlyStoppingError(msg)


class FileLoggingCallback(Callback):
    def __init__(self, searcher, output_dir=None):
        super(FileLoggingCallback, self).__init__()

        self.output_dir = self._prepare_output_dir(output_dir, searcher)

    @staticmethod
    def open(file_path, mode):
        return open(file_path, mode=mode)

    @staticmethod
    def mkdirs(dir_path, exist_ok):
        os.makedirs(dir_path, exist_ok=exist_ok)

    def _prepare_output_dir(self, log_dir, searcher):
        if log_dir is None:
            log_dir = 'log'
        if log_dir[-1] == '/':
            log_dir = log_dir[:-1]

        running_dir = f'exp_{searcher.__class__.__name__}_{datetime.datetime.now().__format__("%m%d-%H%M%S")}'
        output_path = os.path.expanduser(f'{log_dir}/{running_dir}')

        self.mkdirs(output_path, exist_ok=True)
        return output_path

    def on_build_estimator(self, hyper_model, space, estimator, trial_no):
        pass

    def on_trial_begin(self, hyper_model, space, trial_no):
        pass
        # with open(f'{self.output_dir}/trial_{trial_no}.log', 'w') as f:
        #     f.write(space.params_summary())

    def on_trial_end(self, hyper_model, space, trial_no, reward, improved, elapsed):
        with self.open(f'{self.output_dir}/trial_{improved}_{trial_no:04d}_{reward:010.8f}_{elapsed:06.2f}.log',
                       'w') as f:
            f.write(space.params_summary())
            f.write('\r\n----------------Summary for Searcher----------------\r\n')
            f.write(hyper_model.searcher.summary())

        topn = 10
        diff = hyper_model.history.diff(hyper_model.history.get_top(topn))
        with self.open(f'{self.output_dir}/top_{topn}_diff.txt', 'w') as f:
            diff_str = json.dumps(diff, indent=5)
            f.write(diff_str)
            f.write('\r\n')
            f.write(hyper_model.searcher.summary())
        with self.open(f'{self.output_dir}/top_{topn}_config.txt', 'w') as f:
            trials = hyper_model.history.get_top(topn)
            configs = hyper_model.export_configuration(trials)
            for trial, conf in zip(trials, configs):
                f.write(f'Trial No: {trial.trial_no}, Reward: {trial.reward}\r\n')
                f.write(conf)
                f.write('\r\n---------------------------------------------------\r\n\r\n')

    def on_skip_trial(self, hyper_model, space, trial_no, reason, reward, improved, elapsed):
        with self.open(
                f'{self.output_dir}/trial_{reason}_{improved}_{trial_no:04d}_{reward:010.8f}_{elapsed:06.2f}.log',
                'w') as f:
            f.write(space.params_summary())

        topn = 5
        diff = hyper_model.history.diff(hyper_model.history.get_top(topn))
        with self.open(f'{self.output_dir}/top_{topn}_diff.txt', 'w') as f:
            diff_str = json.dumps(diff, indent=5)
            f.write(diff_str)


class FileStorageLoggingCallback(FileLoggingCallback):
    @staticmethod
    def open(file_path, mode):
        return fs.open(file_path, mode=mode)

    @staticmethod
    def mkdirs(dir_path, exist_ok):
        fs.mkdirs(dir_path, exist_ok=exist_ok)


class SummaryCallback(Callback):
    def on_build_estimator(self, hyper_model, space, estimator, trial_no):
        # if logger.is_info_enabled():
        #     logger.info(f'\nTrial No:{trial_no}')
        #     logger.info(space.params_summary())
        estimator.summary()

    def on_trial_begin(self, hyper_model, space, trial_no):
        if logger.is_info_enabled():
            msg = f'\nTrial No:{trial_no}{space.params_summary()}\ntrial {trial_no} begin'
            logger.info(msg)

    def on_trial_end(self, hyper_model, space, trial_no, reward, improved, elapsed):
        if logger.is_info_enabled():
            logger.info(f'trial end. reward:{reward}, improved:{improved}, elapsed:{elapsed}')
            logger.info(f'Total elapsed:{time.time() - hyper_model.start_search_time}')

    def on_skip_trial(self, hyper_model, space, trial_no, reason, reward, improved, elapsed):
        if logger.is_info_enabled():
            logger.info(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            logger.info(f'trial skip. reason:{reason},  reward:{reward}, improved:{improved}, elapsed:{elapsed}')
            logger.info(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
