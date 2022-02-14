# -*- coding:utf-8 -*-
import gc

from .cfg import DispatchCfg as c
from ..core.callbacks import EarlyStoppingError
from ..core.dispatcher import Dispatcher
from ..core.trial import Trial
from ..utils import logging, fs

logger = logging.get_logger(__name__)


class InProcessDispatcher(Dispatcher):
    def __init__(self, models_dir):
        super(InProcessDispatcher, self).__init__()

        self.models_dir = models_dir
        fs.makedirs(models_dir, exist_ok=True)

    def dispatch(self, hyper_model, X, y, X_eval, y_eval, cv, num_folds, max_trials, dataset_id, trial_store,
                 **fit_kwargs):
        retry_limit = c.trial_retry_limit

        trial_no = 1
        retry_counter = 0

        while trial_no <= max_trials:
            gc.collect()
            try:
                space_sample = hyper_model.searcher.sample()
                if hyper_model.history.is_existed(space_sample):
                    if retry_counter >= retry_limit:
                        logger.info(f'Unable to take valid sample and exceed the retry limit {retry_limit}.')
                        break
                    trial = hyper_model.history.get_trial(space_sample)
                    for callback in hyper_model.callbacks:
                        callback.on_skip_trial(hyper_model, space_sample, trial_no, 'trial_existed',
                                               trial.reward, False, trial.elapsed)
                    retry_counter += 1
                    continue

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
                    callback.on_trial_begin(hyper_model, space_sample, trial_no)

                model_file = '%s/%05d_%s.pkl' % (self.models_dir, trial_no, space_sample.space_id)

                trial = hyper_model._run_trial(space_sample, trial_no, X, y, X_eval, y_eval, cv, num_folds, model_file,
                                               **fit_kwargs)

                if trial.succeeded:
                    improved = hyper_model.history.append(trial)
                    for callback in hyper_model.callbacks:
                        callback.on_trial_end(hyper_model, space_sample, trial_no, trial.reward,
                                              improved, trial.elapsed)
                else:
                    hyper_model.history.append(trial)
                    for callback in hyper_model.callbacks:
                        callback.on_trial_error(hyper_model, space_sample, trial_no)

                if logger.is_info_enabled():
                    msg = f'Trial {trial_no} done, reward: {trial.reward}, ' \
                          f'best_trial_no:{hyper_model.best_trial_no}, best_reward:{hyper_model.best_reward}\n'
                    logger.info(msg)
                if trial_store is not None:
                    trial_store.put(dataset_id, trial)
            except EarlyStoppingError:
                break
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

        return trial_no
