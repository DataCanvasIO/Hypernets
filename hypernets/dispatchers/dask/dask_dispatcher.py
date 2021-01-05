# -*- coding:utf-8 -*-

import math
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import dask
from dask.distributed import Client, default_client

from hypernets.core.callbacks import EarlyStoppingError
from hypernets.core.dispatcher import Dispatcher
from hypernets.core.trial import Trail
from hypernets.utils import logging, fs
from hypernets.utils.common import config, Counter

logger = logging.get_logger(__name__)


class DaskTrailItem(Trail):
    def __init__(self, space_sample, trail_no, reward=math.nan, elapsed=math.nan, model_file=None):
        super(DaskTrailItem, self).__init__(space_sample, trail_no,
                                            reward, elapsed,
                                            model_file)

        self.space_id = space_sample.space_id
        self.queue_at = time.time()

    def __str__(self):
        return f'{self.__dict__}'

    @staticmethod
    def copy_from(other):
        return DaskTrailItem(other.space_sample,
                             other.trail_no,
                             other.reward,
                             other.elapsed,
                             other.model_file)


class DaskExecutorPool(object):
    def __init__(self, worker_count, queue_size,
                 on_trail_start, on_trail_done,
                 fn_run_trail,
                 X, y, X_val, y_val, trail_kwargs):
        self.on_trail_start = on_trail_start
        self.on_trail_done = on_trail_done

        self.fn_run_trail_delayed = dask.delayed(fn_run_trail)
        self.X_delayed = dask.delayed(X)
        self.y_delayed = dask.delayed(y)
        self.X_val_delayed = dask.delayed(X_val)
        self.y_val_delayed = dask.delayed(y_val)
        self.trail_kwargs = trail_kwargs

        self.running = False
        self.interrupted = False

        self.worker_count = worker_count
        self.queue_size = queue_size

        self.pool = ThreadPoolExecutor(max_workers=worker_count)
        self.queue = queue.Queue(queue_size)
        self.tasks = []

    @property
    def done(self):
        return all(t.done for t in self.tasks)

    @property
    def qsize(self):
        return self.queue.qsize()

    def push(self, trail_item):
        self.queue.put(trail_item)

    def start(self):
        assert len(self.tasks) == 0

        self.running = True
        self.tasks = [self.pool.submit(self._wait_and_run, self.queue, self._run_trail)
                      for _ in range(self.worker_count)]

        return self.tasks

    def stop(self):
        self.running = False

    def join(self):
        for f in as_completed(self.tasks):
            # logger.info(f'Handle trail count: {f.result()}')
            pass

    def cancel(self):
        self.running = False
        self.pool.shutdown(True)

    def _wait_and_run(self, trail_queue, fn):
        n = 0
        while self.running:
            trail_item = trail_queue.get(block=True)
            if self.running and trail_item is not None:
                fn(trail_item)
                n += 1
            else:
                # invoke others
                self.queue.put(trail_item)
                break

        return n

    def _run_trail(self, trail_item):
        try:
            if self.on_trail_start:
                self.on_trail_start(trail_item)

            fn = self.fn_run_trail_delayed
            d = fn(trail_item.space_sample,
                   trail_item.trail_no,
                   self.X_delayed, self.y_delayed,
                   self.X_val_delayed, self.y_val_delayed,
                   trail_item.model_file,
                   **self.trail_kwargs)
            result = d.compute()

            trail_item.reward = result.reward
            trail_item.elapsed = result.elapsed

            if self.on_trail_done:
                self.on_trail_done(trail_item)

        except KeyboardInterrupt:
            self.running = False
            self.interrupted = True
            self.queue.put(None)  # mark end
            print('KeyboardInterrupt')
        except Exception as e:
            import traceback
            msg = f'{">" * 20} Trail {trail_item.trail_no} failed! {"<" * 20}\n' \
                  + f'{e.__class__.__name__}: {e}\n' \
                  + traceback.format_exc() \
                  + '*' * 50
            logger.error(msg)

            if self.on_trail_done:
                try:
                    self.on_trail_done(trail_item)
                except:
                    pass


class DaskDispatcher(Dispatcher):
    def __init__(self, work_dir):
        try:
            default_client()
        except ValueError:
            # create default Client
            # client = Client("tcp://127.0.0.1:55208")
            # client = Client(processes=False, threads_per_worker=5, n_workers=1, memory_limit='4GB')
            Client()  # detect env: DASK_SCHEDULER_ADDRESS

        super(DaskDispatcher, self).__init__()

        self.work_dir = work_dir
        self.models_dir = f'{work_dir}/models'

        fs.makedirs(self.models_dir, exist_ok=True)

    def dispatch(self, hyper_model, X, y, X_val, y_val, cv, num_folds, max_trails, dataset_id, trail_store,
                 **fit_kwargs):
        assert not any(dask.is_dask_collection(i) for i in (X, y, X_val, y_val)), \
            f'{self.__class__.__name__} does not support to run trail with dask collection.'

        queue_size = int(config('search_queue', '1'))
        worker_count = int(config('search_executors', '3'))
        retry_limit = int(config('search_retry', '1000'))

        failed_counter = Counter()
        success_counter = Counter()

        def on_trail_start(trail_item):
            trail_item.start_at = time.time()
            if logger.is_info_enabled():
                msg = f'Start trail {trail_item.trail_no}, space_id={trail_item.space_id}' \
                      + f',model_file={trail_item.model_file}'
                logger.info(msg)
            for callback in hyper_model.callbacks:
                # callback.on_build_estimator(hyper_model, space_sample, estimator, trail_no) #fixme
                callback.on_trail_begin(hyper_model, trail_item.space_sample, trail_item.trail_no)

        def on_trail_done(trail_item):
            trail_item.done_at = time.time()

            if trail_item.reward != 0 and not math.isnan(trail_item.reward):  # success
                improved = hyper_model.history.append(trail_item)
                for callback in hyper_model.callbacks:
                    callback.on_trail_end(hyper_model, trail_item.space_sample,
                                          trail_item.trail_no, trail_item.reward,
                                          improved, trail_item.elapsed)
                success_counter()
            else:
                for callback in hyper_model.callbacks:
                    callback.on_trail_error(hyper_model, trail_item.space_sample, trail_item.trail_no)
                failed_counter()

            if logger.is_info_enabled():
                elapsed = '%.3f' % (trail_item.done_at - trail_item.start_at)
                msg = f'Trail {trail_item.trail_no} done with reward={trail_item.reward}, ' \
                      f'elapsed {elapsed} seconds\n'
                logger.info(msg)
            if trail_store is not None:
                trail_store.put(dataset_id, trail_item)

        pool = DaskExecutorPool(worker_count, queue_size,
                                on_trail_start, on_trail_done,
                                hyper_model._run_trial,
                                X, y, X_val, y_val, fit_kwargs)
        pool.start()

        trail_no = 1
        retry_counter = 0

        while trail_no <= max_trails and pool.running:
            if pool.qsize >= queue_size:
                time.sleep(0.1)
                continue

            space_sample = hyper_model.searcher.sample()
            if hyper_model.history.is_existed(space_sample):
                if retry_counter >= retry_limit:
                    logger.info(f'Unable to take valid sample and exceed the retry limit 1000.')
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

                model_file = '%s/%05d_%s.pkl' % (self.models_dir, trail_no, space_sample.space_id)

                item = DaskTrailItem(space_sample, trail_no, model_file=model_file)
                pool.push(item)

                if logger.is_info_enabled():
                    logger.info(f'Found trail {trail_no}, queue size: {pool.qsize}')
            except EarlyStoppingError:
                pool.stop()
                break
            except KeyboardInterrupt:
                pool.stop()
                pool.interrupted = True
                print('KeyboardInterrupt')
                break
            except Exception as e:
                import traceback
                msg = f'{">" * 20} Search trail {trail_no} failed! {"<" * 20}\n' \
                      + f'{e.__class__.__name__}: {e}\n' \
                      + traceback.format_exc() \
                      + '*' * 50
                logger.error(msg)
            finally:
                trail_no += 1
                retry_counter = 0

        # wait trails
        if pool.running:
            logger.info('Search done, wait trail tasks.')
        pool.push(None)  # mark end
        pool.join()

        if logger.is_info_enabled():
            logger.info(f'Search and all trails done, {success_counter.value} success, '
                        f'{failed_counter.value} failed.')

        return trail_no
