# -*- coding:utf-8 -*-

import time

from ..core.callbacks import EarlyStoppingError
from ..core.dispatcher import Dispatcher
from ..core.trial import Trail
from ..utils import logging, fs
from ..utils.common import config

logger = logging.get_logger(__name__)

_model_root = config('model_path', 'tmp/models')


class InProcessDispatcher(Dispatcher):
    def __init__(self, models_dir):
        super(InProcessDispatcher, self).__init__()

        self.models_dir = models_dir
        fs.makedirs(models_dir, exist_ok=True)

    def dispatch(self, hyper_model, X, y, X_val, y_val, max_trails, dataset_id, trail_store,
                 **fit_kwargs):
        retry_limit = int(config('search_retry', '1000'))

        trail_no = 1
        retry_counter = 0
        while trail_no <= max_trails:
            space_sample = hyper_model.searcher.sample()
            if hyper_model.history.is_existed(space_sample):
                if retry_counter >= retry_limit:
                    logger.info(f'Unable to take valid sample and exceed the retry limit {retry_limit}.')
                    break
                trail = hyper_model.history.get_trail(space_sample)
                for callback in hyper_model.callbacks:
                    callback.on_skip_trail(hyper_model, space_sample, trail_no, 'trail_existed', trail.reward, False,
                                           trail.elapsed)
                retry_counter += 1
                continue

            try:
                if trail_store is not None:
                    trail = trail_store.get(dataset_id, space_sample)
                    if trail is not None:
                        reward = trail.reward
                        elapsed = trail.elapsed
                        trail = Trail(space_sample, trail_no, reward, elapsed)
                        improved = hyper_model.history.append(trail)
                        hyper_model.searcher.update_result(space_sample, reward)
                        for callback in hyper_model.callbacks:
                            callback.on_skip_trail(hyper_model, space_sample, trail_no, 'hit_trail_store', reward,
                                                   improved,
                                                   elapsed)
                        trail_no += 1
                        continue

                for callback in hyper_model.callbacks:
                    # callback.on_build_estimator(hyper_model, space_sample, estimator, trail_no) #fixme
                    callback.on_trail_begin(hyper_model, space_sample, trail_no)

                model_file = '%s/%05d_%s.pkl' % (self.models_dir, trail_no, space_sample.space_id)

                start_at = time.time()
                trail = hyper_model._run_trial(space_sample, trail_no, X, y, X_val, y_val, model_file, **fit_kwargs)
                done_at = time.time()
                logger.info('*' * 20 + ' elapsed %.3f seconds' % (done_at - start_at))

                if trail.reward != 0:  # success
                    improved = hyper_model.history.append(trail)
                    for callback in hyper_model.callbacks:
                        callback.on_trail_end(hyper_model, space_sample, trail_no, trail.reward,
                                              improved, trail.elapsed)
                else:
                    for callback in hyper_model.callbacks:
                        callback.on_trail_error(hyper_model, space_sample, trail_no)

                if logger.is_info_enabled():
                    msg = f'Trail {trail_no} done\n' \
                          + f'----------------------------------------------------------------\n' \
                          + f'space signatures: \n{hyper_model.history.get_space_signatures()}\n' \
                          + f'----------------------------------------------------------------'
                    logger.info(msg)
                if trail_store is not None:
                    trail_store.put(dataset_id, trail)
            except EarlyStoppingError:
                break
                # TODO: early stopping
            except Exception as e:
                import sys
                import traceback
                msg = f'{">" * 20} Trail {trail_no} failed! {"<" * 20}\n' \
                      + f'{e.__class__.__name__}: {e}\n' \
                      + traceback.format_exc() \
                      + '*' * 50
                logger.error(msg)
            finally:
                trail_no += 1
                retry_counter = 0

        return trail_no
