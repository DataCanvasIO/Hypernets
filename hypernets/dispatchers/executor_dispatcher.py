# -*- coding:utf-8 -*-
import copy
import pickle
import sys
import time

from ..core.dispatcher import Dispatcher
from ..dispatchers.grpc.search_driver_client import SearchDriverClient
from ..utils import logging

logger = logging.get_logger(__name__)

_search_counter = 0


class ExecutorDispatcher(Dispatcher):
    def __init__(self, driver_address):
        super(ExecutorDispatcher, self).__init__()
        self.driver_address = driver_address

    def dispatch(self, hyper_model, X, y, X_val, y_val, max_trails, dataset_id, trail_store,
                 **fit_kwargs):

        if 'search_id' in fit_kwargs:
            search_id = fit_kwargs.pop('search_id')
        else:
            global _search_counter
            _search_counter += 1
            search_id = 'search-%02d' % _search_counter

        if logger.is_info_enabled():
            logger.info(f'[{search_id}] started')
            print(f'[{search_id}] connect to driver {self.driver_address}', end='')
        client = SearchDriverClient(self.driver_address, search_id)
        client.ping(wait=True)

        def response(item_x, success, reward=0.0, message=''):
            res = copy.copy(item_x)
            res.success = success
            res.reward = reward
            res.message = message
            return res

        trail_no = 0
        sch = client.search(search_id)
        try:
            item = next(sch)
            while item:
                if item.is_waiting():
                    if logger.is_info_enabled():
                        logger.info(f'[{search_id}] not found search, wait and continue')
                    time.sleep(1)
                    item = sch.send(response(item, True))
                    continue

                if item.is_finished():
                    if logger.is_info_enabled():
                        logger.info(f'[{search_id}] search finished, exit.')
                    # sch.send(None)
                    break

                if not item.is_ok():
                    if logger.is_info_enabled():
                        logger.info(f'[{search_id}] dispatched with {item.code}, exit.')
                    # sch.send(None)
                    break

                trail_no = item.trail_no if item.trail_no is not None else trail_no + 1
                detail = f'trail_no={trail_no}, space_id={item.space_id}, space_file={item.space_file}'
                if logger.is_info_enabled():
                    logger.info(f'[{search_id}] new trail:' + detail)
                try:
                    with open(item.space_file, 'rb') as f:
                        space_sample = pickle.load(f)

                    for callback in hyper_model.callbacks:
                        # callback.on_build_estimator(hyper_model, space_sample, estimator, trail_no)
                        callback.on_trail_begin(hyper_model, space_sample, trail_no)

                    trail = hyper_model._run_trial(space_sample, trail_no, X, y, X_val, y_val, **fit_kwargs)
                    if trail.reward != 0:
                        improved = hyper_model.history.append(trail)
                        if improved:
                            hyper_model.best_model = hyper_model.last_model

                        for callback in hyper_model.callbacks:
                            callback.on_trail_end(hyper_model, space_sample, trail_no, trail.reward,
                                                  improved, trail.elapsed)
                    else:
                        for callback in hyper_model.callbacks:
                            callback.on_trail_error(hyper_model, space_sample, trail_no)

                    if logger.is_info_enabled():
                        logger.info(f'----------------------------------------------------------------')
                        logger.info(f'space signatures: \n{hyper_model.history.get_space_signatures()}')
                        logger.info(f'----------------------------------------------------------------')
                    if trail_store is not None:
                        trail_store.put(dataset_id, trail)

                    item = sch.send(response(item, trail.reward != 0.0, trail.reward))
                except StopIteration:
                    break
                except KeyboardInterrupt:
                    if logger.is_info_enabled():
                        logger.info('KeyboardInterrupt')
                    break
                except Exception as e:
                    import traceback
                    msg = f'[{search_id}] {e.__class__.__name__}: {e}'
                    logger.error(msg + '\n' + traceback.format_exc())
                    item = sch.send(response(item, False, 0.0, msg))
        except StopIteration as e:
            pass
        finally:
            sch.close()
            client.close()

        if logger.is_info_enabled():
            logger.info(f'[{search_id}] search done, last trail_no={trail_no}')
        return trail_no
