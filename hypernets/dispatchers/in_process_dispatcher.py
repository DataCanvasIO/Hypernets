# -*- coding:utf-8 -*-

import time

import pandas as pd
from IPython.display import display, update_display, display_markdown

from ..core.callbacks import EarlyStoppingError
from ..core.dispatcher import Dispatcher
from ..core.trial import Trial
from ..utils import logging, fs
from ..utils.common import config, isnotebook

logger = logging.get_logger(__name__)

_is_notebook = isnotebook()
_model_root = config('model_path', 'tmp/models')


class InProcessDispatcher(Dispatcher):
    def __init__(self, models_dir):
        super(InProcessDispatcher, self).__init__()

        self.models_dir = models_dir
        fs.makedirs(models_dir, exist_ok=True)

    def dispatch(self, hyper_model, X, y, X_eval, y_eval, cv, num_folds, max_trials, dataset_id, trial_store,
                 **fit_kwargs):
        retry_limit = int(config('search_retry', '1000'))

        trial_no = 1
        retry_counter = 0
        current_trial_display_id = None
        search_summary_display_id = None
        best_trial_display_id = None
        title_display_id = None
        start_time = time.time()
        last_reward = 0
        while trial_no <= max_trials:
            space_sample = hyper_model.searcher.sample()
            if hyper_model.history.is_existed(space_sample):
                if retry_counter >= retry_limit:
                    logger.info(f'Unable to take valid sample and exceed the retry limit {retry_limit}.')
                    break
                trial = hyper_model.history.get_trial(space_sample)
                for callback in hyper_model.callbacks:
                    callback.on_skip_trial(hyper_model, space_sample, trial_no, 'trial_existed', trial.reward, False,
                                           trial.elapsed)
                retry_counter += 1
                continue

            try:
                if trial_store is not None:
                    trial = trial_store.get(dataset_id, space_sample)
                    if trial is not None:
                        reward = trial.reward
                        elapsed = trial.elapsed
                        trial = Trial(space_sample, trial_no, reward, elapsed)
                        improved = hyper_model.history.append(trial)
                        hyper_model.searcher.update_result(space_sample, reward)
                        for callback in hyper_model.callbacks:
                            callback.on_skip_trial(hyper_model, space_sample, trial_no, 'hit_trial_store', reward,
                                                   improved,
                                                   elapsed)
                        trial_no += 1
                        continue

                for callback in hyper_model.callbacks:
                    # callback.on_build_estimator(hyper_model, space_sample, estimator, trial_no) #fixme
                    callback.on_trial_begin(hyper_model, space_sample, trial_no)

                model_file = '%s/%05d_%s.pkl' % (self.models_dir, trial_no, space_sample.space_id)

                if _is_notebook:
                    df_summary = pd.DataFrame([(trial_no, last_reward, hyper_model.best_trial_no,
                                                hyper_model.best_reward,
                                                time.time() - start_time, max_trials)],
                                              columns=['trial No.', 'Previous reward', 'Best trial', 'Best reward',
                                                       'Total elapsed',
                                                       'Max trials'])
                    if search_summary_display_id is None:
                        handle = display(df_summary, display_id=True)
                        if handle is not None:
                            search_summary_display_id = handle.display_id
                    else:
                        update_display(df_summary, display_id=search_summary_display_id)

                    if current_trial_display_id is None:
                        handle = display({'text/markdown': '#### Current Trial:'}, raw=True, include=['text/markdown'],
                                         display_id=True)
                        if handle is not None:
                            title_display_id = handle.display_id
                        handle = display(space_sample, display_id=True)
                        if handle is not None:
                            current_trial_display_id = handle.display_id
                    else:
                        update_display(space_sample, display_id=current_trial_display_id)

                trial = hyper_model._run_trial(space_sample, trial_no, X, y, X_eval, y_eval, cv, num_folds, model_file,
                                               **fit_kwargs)
                last_reward = trial.reward
                if trial.reward != 0:  # success
                    improved = hyper_model.history.append(trial)
                    for callback in hyper_model.callbacks:
                        callback.on_trial_end(hyper_model, space_sample, trial_no, trial.reward,
                                              improved, trial.elapsed)
                else:
                    for callback in hyper_model.callbacks:
                        callback.on_trial_error(hyper_model, space_sample, trial_no)

                if _is_notebook:
                    best_trial = hyper_model.get_best_trial()
                    if best_trial is not None:
                        if best_trial_display_id is None:
                            display_markdown('#### Best Trial:', raw=True)
                            handle = display(best_trial.space_sample, display_id=True)
                            if handle is not None:
                                best_trial_display_id = handle.display_id
                        else:
                            update_display(best_trial.space_sample, display_id=best_trial_display_id)

                if logger.is_info_enabled():
                    msg = f'Trial {trial_no} done, reward: {trial.reward}, ' \
                          f'best_trial_no:{hyper_model.best_trial_no}, best_reward:{hyper_model.best_reward}\n'
                    logger.info(msg)
                if trial_store is not None:
                    trial_store.put(dataset_id, trial)
            except EarlyStoppingError:
                break
                # TODO: early stopping
            except Exception as e:
                import sys
                import traceback
                msg = f'{">" * 20} Trial {trial_no} failed! {"<" * 20}\n' \
                      + f'{e.__class__.__name__}: {e}\n' \
                      + traceback.format_exc() \
                      + '*' * 50
                logger.error(msg)
            finally:
                trial_no += 1
                retry_counter = 0

        if _is_notebook:
            update_display({'text/markdown': '#### Top trials:'}, raw=True, include=['text/markdown'],
                           display_id=title_display_id)
            df_best_trials = pd.DataFrame([
                (t.trial_no, t.reward, t.elapsed, t.space_sample.vectors) for t in hyper_model.get_top_trials(5)],
                columns=['Trial No.', 'Reward', 'Elapsed', 'Space Vector'])
            if current_trial_display_id is None:
                display(df_best_trials, display_id=True)
            else:
                update_display(df_best_trials, display_id=current_trial_display_id)

        return trial_no
