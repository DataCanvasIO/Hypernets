# -*- coding:utf-8 -*-
import gc

from .cfg import DispatchCfg as c
from ..core.callbacks import EarlyStoppingError
from ..core.dispatcher import Dispatcher
from ..core.trial import Trial
from ..tabular import get_tool_box
from ..utils import logging, fs, const

logger = logging.get_logger(__name__)


class InProcessDispatcher(Dispatcher):
    def __init__(self, models_dir):
        super(InProcessDispatcher, self).__init__()

        self.models_dir = models_dir
        fs.makedirs(models_dir, exist_ok=True)

    def dispatch(self, hyper_model, X, y, X_eval, y_eval, X_test, cv, num_folds, max_trials, dataset_id, trial_store,
                 **fit_kwargs):
        retry_limit = c.trial_retry_limit

        trial_no = 1
        retry_counter = 0

        space_options = {}
        if hyper_model.searcher.kind() == const.SEARCHER_MOO:
            if 'feature_usage' in [_.name for _ in hyper_model.searcher.objectives]:
                tb = get_tool_box(X, y)
                preprocessor = tb.general_preprocessor(X)
                estimator = tb.general_estimator(X, y, task=hyper_model.task)
                estimator.fit(preprocessor.fit_transform(X, y), y)
                importances = list(zip(estimator.feature_name_, estimator.feature_importances_))
                space_options['importances'] = importances

        while trial_no <= max_trials:
            gc.collect()
            try:

                space_sample = hyper_model.searcher.sample(space_options=space_options)
                if hyper_model.history.is_existed(space_sample):
                    if retry_counter >= retry_limit:
                        logger.info(f'Unable to take valid sample and exceed the retry limit {retry_limit}.')
                        break
                    trial = hyper_model.history.get_trial(space_sample)
                    for callback in hyper_model.callbacks:
                        try:
                            callback.on_skip_trial(hyper_model, space_sample, trial_no, 'trial_existed',
                                                   trial.reward, False, trial.elapsed)
                        except EarlyStoppingError:
                            raise
                        except Exception as e:
                            logger.warn(e)

                    retry_counter += 1
                    continue

                if trial_store is not None:
                    trial_hit = trial_store.get(dataset_id, space_sample)
                    if trial_hit is not None and fs.exists(trial_hit.model_file):
                        reward = trial_hit.reward
                        elapsed = trial_hit.elapsed
                        trial = Trial(space_sample, trial_no,
                                      reward=reward,
                                      elapsed=elapsed,
                                      model_file=trial_hit.model_file,
                                      succeeded=trial_hit.succeeded)
                        trial.memo = trial_hit.memo.copy()
                        trial.iteration_scores = trial_hit.iteration_scores.copy()

                        improved = hyper_model.history.append(trial)
                        hyper_model.searcher.update_result(space_sample, reward)
                        for callback in hyper_model.callbacks:
                            try:
                                callback.on_skip_trial(hyper_model, space_sample, trial_no, 'hit_trial_store',
                                                       reward, improved, elapsed)
                            except EarlyStoppingError:
                                raise
                            except Exception as e:
                                logger.warn(e)
                        continue

                for callback in hyper_model.callbacks:
                    try:
                        callback.on_trial_begin(hyper_model, space_sample, trial_no)
                    except EarlyStoppingError:
                        raise
                    except Exception as e:
                        logger.warn(e)

                model_file = '%s/%05d_%s.pkl' % (self.models_dir, trial_no, space_sample.space_id)

                trial = hyper_model._run_trial(space_sample, trial_no, X, y, X_eval, y_eval, X_test, cv, num_folds, model_file,
                                               **fit_kwargs)

                if trial.succeeded:
                    improved = hyper_model.history.append(trial)
                    for callback in hyper_model.callbacks:
                        try:
                            callback.on_trial_end(hyper_model, space_sample, trial_no, trial.reward,
                                                  improved, trial.elapsed)
                        except EarlyStoppingError:
                            raise
                        except Exception as e:
                            logger.warn(e)

                else:
                    hyper_model.history.append(trial)
                    for callback in hyper_model.callbacks:
                        try:
                            callback.on_trial_error(hyper_model, space_sample, trial_no)
                        except EarlyStoppingError:
                            raise
                        except Exception as e:
                            logger.warn(e)

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
