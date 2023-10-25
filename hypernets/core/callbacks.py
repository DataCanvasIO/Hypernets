# -*- coding:utf-8 -*-
"""

"""
import datetime
import json
import os
import time

import numpy as np
import pandas as pd
from IPython.display import display, update_display, display_markdown
from tqdm.auto import tqdm

from ..utils import logging, fs, to_repr

logger = logging.get_logger(__name__)


class Callback():
    def __init__(self):
        pass

    def on_search_start(self, hyper_model, X, y, X_eval, y_eval, cv, num_folds, max_trials, dataset_id, trial_store,
                        **fit_kwargs):
        pass

    def on_search_end(self, hyper_model):
        pass

    def on_search_error(self, hyper_model):
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

    def __repr__(self):
        return to_repr(self)


class EarlyStoppingError(RuntimeError):
    def __init__(self, *arg):
        self.args = arg


class EarlyStoppingCallback(Callback):
    REASON_TRIAL_LIMIT = 'max_no_improvement_trials'
    REASON_TIME_LIMIT = 'time_limit'
    REASON_EXPECTED_REWARD = 'expected_reward'

    def __init__(self, max_no_improvement_trials=0, mode='min', min_delta=0, time_limit=None, expected_reward=None):
        super(Callback, self).__init__()
        # assert time_limit is None or time_limit > 60, 'If `time_limit` is not None, it must be greater than 60.'

        # settings
        if mode == 'min':
            self.op = np.less
        elif mode == 'max':
            self.op = np.greater
        else:
            raise ValueError(f'Unsupported mode:{mode}')
        self.max_no_improvement_trials = max_no_improvement_trials
        self.mode = mode
        self.min_delta = min_delta
        self.time_limit = time_limit
        self.expected_reward = expected_reward

        # running state
        self.start_time = None
        self.best_reward = None
        self.best_trial_no = None
        self.counter_no_improvement_trials = 0
        self.triggered = None
        self.triggered_reason = None

    def on_search_start(self, hyper_model, X, y, X_eval, y_eval, cv, num_folds, max_trials, dataset_id, trial_store,
                        **fit_kwargs):
        self.triggered = False
        self.triggered_reason = None

    def on_trial_begin(self, hyper_model, space, trial_no):
        if self.start_time is None:
            self.start_time = time.time()

    def on_trial_end(self, hyper_model, space, trial_no, reward, improved, elapsed):
        reward = reward[0]  # NOTE only use first metric

        if self.start_time is None:
            self.start_time = time.time()

        time_total = time.time() - self.start_time

        if self.time_limit is not None and self.time_limit > 0:
            if time_total > self.time_limit:
                self.triggered = True
                self.triggered_reason = self.REASON_TIME_LIMIT

        if self.expected_reward is not None and self.expected_reward != 0.0:
            if self.op(reward, self.expected_reward):
                self.triggered = True
                self.triggered_reason = self.REASON_EXPECTED_REWARD

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
                        self.triggered = True
                        self.triggered_reason = self.REASON_TRIAL_LIMIT

        if self.triggered:
            msg = f'Early stopping on trial : {trial_no}, reason: {self.triggered_reason}, ' \
                  f'best reward: {self.best_reward}, best trial: {self.best_trial_no}, ' \
                  f'elapsed seconds: {time_total}'
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
        reward = reward[0]
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
        reward_repr = "_".join(list(map(lambda v: f"{v:010.8f}", reward)))
        with self.open(
                f'{self.output_dir}/trial_{reason}_{improved}_{trial_no:04d}_{reward_repr}_{elapsed:06.2f}.log',
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
    def __init__(self):
        super(SummaryCallback, self).__init__()

        self.start_search_time = None

    def on_search_start(self, hyper_model, X, y, X_eval, y_eval, cv, num_folds, max_trials, dataset_id, trial_store,
                        **fit_kwargs):
        self.start_search_time = time.time()

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
            logger.info(f'Total elapsed:{time.time() - self.start_search_time}')

    def on_skip_trial(self, hyper_model, space, trial_no, reason, reward, improved, elapsed):
        if logger.is_info_enabled():
            logger.info(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            logger.info(f'trial skip. reason:{reason},  reward:{reward}, improved:{improved}, elapsed:{elapsed}')
            logger.info(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')


class NotebookCallback(Callback):
    def __init__(self):
        super(NotebookCallback, self).__init__()

        self.current_trial_display_id = None
        self.search_summary_display_id = None
        self.best_trial_display_id = None
        self.title_display_id = None

        self.last_trial_no = 0
        self.last_reward = 0
        self.start_time = 0
        self.max_trials = 0

    def on_search_start(self, hyper_model, X, y, X_eval, y_eval, cv, num_folds, max_trials, dataset_id, trial_store,
                        **fit_kwargs):
        self.start_time = time.time()
        self.max_trials = max_trials

        df_holder = pd.DataFrame()
        settings = {'X': X.shape,
                    'y': y.shape,
                    'X_eval': X_eval.shape if X_eval is not None else None,
                    'y_eval': y_eval.shape if y_eval is not None else None,
                    'cv': cv,
                    'num_folds': num_folds,
                    'max_trials': max_trials,
                    # 'dataset_id': dataset_id,
                    # 'trail_store': trial_store,
                    'fit_kwargs': fit_kwargs.keys()
                    }
        df_settings = pd.DataFrame({k: [v] for k, v in settings.items()})

        display_markdown('#### Experiment Settings:', raw=True)
        display(hyper_model, display_id=False)
        display(df_settings, display_id=False)

        display_markdown('#### Trials Summary:', raw=True)
        handle = display(df_holder, display_id=True)
        if handle is not None:
            self.search_summary_display_id = handle.display_id

        display_markdown('#### Best Trial:', raw=True)
        handle = display(df_holder, display_id=True)
        if handle is not None:
            self.best_trial_display_id = handle.display_id

        handle = display({'text/markdown': '#### Current Trial:'}, raw=True, include=['text/markdown'],
                         display_id=True)
        if handle is not None:
            self.title_display_id = handle.display_id

        handle = display(df_holder, display_id=True)
        if handle is not None:
            self.current_trial_display_id = handle.display_id

    def on_trial_begin(self, hyper_model, space, trial_no):
        df_summary = pd.DataFrame([(trial_no, self.last_reward, hyper_model.best_trial_no,
                                    hyper_model.best_reward,
                                    time.time() - self.start_time,
                                    len([t for t in hyper_model.history.trials if t.succeeded]),
                                    self.max_trials)],
                                  columns=['Trial No.', 'Previous reward', 'Best trial', 'Best reward',
                                           'Total elapsed', 'Valid trials',
                                           'Max trials'])
        if self.search_summary_display_id is not None:
            update_display(df_summary, display_id=self.search_summary_display_id)

        if self.current_trial_display_id is not None:
            update_display(space, display_id=self.current_trial_display_id)

    def on_search_end(self, hyper_model):
        df_summary = pd.DataFrame([(self.last_trial_no, self.last_reward, hyper_model.best_trial_no,
                                    hyper_model.best_reward,
                                    time.time() - self.start_time,
                                    len([t for t in hyper_model.history.trials if t.succeeded]),
                                    self.max_trials)],
                                  columns=['Trial No.', 'Previous reward', 'Best trial', 'Best reward',
                                           'Total elapsed', 'Valid trials',
                                           'Max trials'])
        if self.search_summary_display_id is not None:
            update_display(df_summary, display_id=self.search_summary_display_id)

        if self.title_display_id is not None:
            update_display({'text/markdown': '#### Top trials:'}, raw=True, include=['text/markdown'],
                           display_id=self.title_display_id)

        df_best_trials = pd.DataFrame([
            (t.trial_no, t.reward, t.elapsed, t.space_sample.vectors) for t in hyper_model.get_top_trials(5)],
            columns=['Trial No.', 'Reward', 'Elapsed', 'Space Vector'])
        if self.current_trial_display_id is not None:
            update_display(df_best_trials, display_id=self.current_trial_display_id)

    def on_trial_end(self, hyper_model, space, trial_no, reward, improved, elapsed):
        reward = reward[0]
        self.last_trial_no = trial_no
        self.last_reward = reward

        best_trial = hyper_model.get_best_trial()
        if best_trial is not None and not isinstance(best_trial, list) and self.best_trial_display_id is not None:
            update_display(best_trial.space_sample, display_id=self.best_trial_display_id)

    def on_trial_error(self, hyper_model, space, trial_no):
        self.last_trial_no = trial_no
        self.last_reward = 'ERR'


class ProgressiveCallback(Callback):
    def __init__(self):
        super(ProgressiveCallback, self).__init__()

        self.pbar = None

    def on_search_start(self, hyper_model, X, y, X_eval, y_eval, cv, num_folds, max_trials, dataset_id, trial_store,
                        **fit_kwargs):
        self.pbar = tqdm(total=max_trials, leave=False, desc='search')

    def on_search_end(self, hyper_model):
        if self.pbar is not None:
            self.pbar.update(self.pbar.total)
            self.pbar.close()
            self.pbar = None

    def on_search_error(self, hyper_model):
        self.on_search_end(hyper_model)

    def on_trial_end(self, hyper_model, space, trial_no, reward, improved, elapsed):
        if self.pbar is not None:
            self.pbar.update(1)

    def on_trial_error(self, hyper_model, space, trial_no):
        if self.pbar is not None:
            self.pbar.update(1)

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__

        state = state.copy()
        state['pbar'] = None

        return state
