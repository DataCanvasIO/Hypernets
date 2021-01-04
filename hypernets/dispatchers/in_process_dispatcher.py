# -*- coding:utf-8 -*-

from ..core.callbacks import EarlyStoppingError
from ..core.dispatcher import Dispatcher
from ..core.trial import Trail
from ..utils import logging, fs
from ..utils.common import config

from IPython.display import display, update_display, clear_output, display_markdown
import time
import pandas as pd

logger = logging.get_logger(__name__)

_model_root = config('model_path', 'tmp/models')


class InProcessDispatcher(Dispatcher):
    def __init__(self, models_dir):
        super(InProcessDispatcher, self).__init__()

        self.models_dir = models_dir
        fs.makedirs(models_dir, exist_ok=True)

    def dispatch(self, hyper_model, X, y, X_eval, y_eval, max_trails, dataset_id, trail_store,
                 **fit_kwargs):
        retry_limit = int(config('search_retry', '1000'))

        trail_no = 1
        retry_counter = 0
        current_trail_display_id = None
        search_summary_display_id = None
        best_trail_display_id = None
        title_display_id = None
        start_time = time.time()
        last_reward = 0
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

                df_summary = pd.DataFrame([(trail_no, last_reward, hyper_model.best_trail_no, hyper_model.best_reward,
                                            time.time() - start_time, max_trails)],
                                          columns=['trail No.', 'Previous reward', 'Best trail', 'Best reward',
                                                   'Total elapsed',
                                                   'Max trails'])
                if search_summary_display_id is None:
                    handle = display(df_summary, display_id=True)
                    if handle is not None:
                        search_summary_display_id = handle.display_id
                else:
                    update_display(df_summary, display_id=search_summary_display_id)

                if current_trail_display_id is None:
                    handle = display({'text/markdown': '#### Current Trail:'}, raw=True, include=['text/markdown'],
                                     display_id=True)
                    if handle is not None:
                        title_display_id = handle.display_id
                    handle = display(space_sample, display_id=True)
                    if handle is not None:
                        current_trail_display_id = handle.display_id
                else:
                    update_display(space_sample, display_id=current_trail_display_id)

                trail = hyper_model._run_trial(space_sample, trail_no, X, y, X_eval, y_eval, model_file, **fit_kwargs)
                last_reward = trail.reward
                if trail.reward != 0:  # success
                    improved = hyper_model.history.append(trail)
                    for callback in hyper_model.callbacks:
                        callback.on_trail_end(hyper_model, space_sample, trail_no, trail.reward,
                                              improved, trail.elapsed)
                else:
                    for callback in hyper_model.callbacks:
                        callback.on_trail_error(hyper_model, space_sample, trail_no)

                best_trail = hyper_model.get_best_trail()
                if best_trail is not None:
                    if best_trail_display_id is None:
                        display_markdown('#### Best Trail:', raw=True)
                        handle = display(best_trail.space_sample, display_id=True)
                        if handle is not None:
                            best_trail_display_id = handle.display_id
                    else:
                        update_display(best_trail.space_sample, display_id=best_trail_display_id)

                if logger.is_info_enabled():
                    msg = f'Trail {trail_no} done, reward: {trail.reward}, best_trail_no:{hyper_model.best_trail_no}, best_reward:{hyper_model.best_reward}\n'
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

        update_display({'text/markdown': '#### Top trails:'}, raw=True, include=['text/markdown'],
                       display_id=title_display_id)

        df_best_trails = pd.DataFrame([
            (t.trail_no, t.reward, t.elapsed, t.space_sample.vectors) for t in hyper_model.get_top_trails(5)],
            columns=['Trail No.', 'Reward', 'Elapsed','Space Vector'])
        if current_trail_display_id is None:
            display(df_best_trails, display_id=True)
        else:
            update_display(df_best_trails, display_id=current_trail_display_id)

        return trail_no
