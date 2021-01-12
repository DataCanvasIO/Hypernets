# -*- coding:utf-8 -*-

import math
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import dask
from dask.distributed import Client, default_client

from hypernets.core.callbacks import EarlyStoppingError
from hypernets.core.dispatcher import Dispatcher
from hypernets.core.trial import Trial
from hypernets.utils import logging, fs
from hypernets.utils.common import config, Counter

logger = logging.get_logger(__name__)


class DaskTrialItem(Trial):
    def __init__(self, space_sample, trial_no, reward=math.nan, elapsed=math.nan, model_file=None):
        super(DaskTrialItem, self).__init__(space_sample, trial_no,
                                            reward, elapsed,
                                            model_file)

        self.space_id = space_sample.space_id
        self.queue_at = time.time()

    def __str__(self):
        return f'{self.__dict__}'

    @staticmethod
    def copy_from(other):
        return DaskTrialItem(other.space_sample,
                             other.trial_no,
                             other.reward,
                             other.elapsed,
                             other.model_file)


class DaskExecutorPool(object):
    def __init__(self, worker_count, queue_size,
                 on_trial_start, on_trial_done,
                 fn_run_trial,
                 X, y, X_val, y_val, trial_kwargs):
        self.on_trial_start = on_trial_start
        self.on_trial_done = on_trial_done

        self.fn_run_trial_delayed = dask.delayed(fn_run_trial)
        self.X_delayed = dask.delayed(X)
        self.y_delayed = dask.delayed(y)
        self.X_val_delayed = dask.delayed(X_val)
        self.y_val_delayed = dask.delayed(y_val)
        self.trial_kwargs = trial_kwargs

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

    def push(self, trial_item):
        self.queue.put(trial_item)

    def start(self):
        assert len(self.tasks) == 0

        self.running = True
        self.tasks = [self.pool.submit(self._wait_and_run, self.queue, self._run_trial)
                      for _ in range(self.worker_count)]

        return self.tasks

    def stop(self):
        self.running = False

    def join(self):
        for f in as_completed(self.tasks):
            # logger.info(f'Handle trial count: {f.result()}')
            pass

    def cancel(self):
        self.running = False
        self.pool.shutdown(True)

    def _wait_and_run(self, trial_queue, fn):
        n = 0
        while self.running:
            trial_item = trial_queue.get(block=True)
            if self.running and trial_item is not None:
                fn(trial_item)
                n += 1
            else:
                # invoke others
                self.queue.put(trial_item)
                break

        return n

    def _run_trial(self, trial_item):
        try:
            if self.on_trial_start:
                self.on_trial_start(trial_item)

            fn = self.fn_run_trial_delayed
            d = fn(trial_item.space_sample,
                   trial_item.trial_no,
                   self.X_delayed, self.y_delayed,
                   self.X_val_delayed, self.y_val_delayed,
                   trial_item.model_file,
                   **self.trial_kwargs)
            result = d.compute()

            trial_item.reward = result.reward
            trial_item.elapsed = result.elapsed

            if self.on_trial_done:
                self.on_trial_done(trial_item)

        except KeyboardInterrupt:
            self.running = False
            self.interrupted = True
            self.queue.put(None)  # mark end
            print('KeyboardInterrupt')
        except Exception as e:
            import traceback
            msg = f'{">" * 20} Trial {trial_item.trial_no} failed! {"<" * 20}\n' \
                  + f'{e.__class__.__name__}: {e}\n' \
                  + traceback.format_exc() \
                  + '*' * 50
            logger.error(msg)

            if self.on_trial_done:
                try:
                    self.on_trial_done(trial_item)
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

    def dispatch(self, hyper_model, X, y, X_val, y_val, cv, num_folds, max_trials, dataset_id, trial_store,
                 **fit_kwargs):
        assert not any(dask.is_dask_collection(i) for i in (X, y, X_val, y_val)), \
            f'{self.__class__.__name__} does not support to run trial with dask collection.'

        queue_size = int(config('search_queue', '1'))
        worker_count = int(config('search_executors', '3'))
        retry_limit = int(config('search_retry', '1000'))

        failed_counter = Counter()
        success_counter = Counter()

        def on_trial_start(trial_item):
            trial_item.start_at = time.time()
            if logger.is_info_enabled():
                msg = f'Start trial {trial_item.trial_no}, space_id={trial_item.space_id}' \
                      + f',model_file={trial_item.model_file}'
                logger.info(msg)
            for callback in hyper_model.callbacks:
                # callback.on_build_estimator(hyper_model, space_sample, estimator, trial_no) #fixme
                callback.on_trial_begin(hyper_model, trial_item.space_sample, trial_item.trial_no)

        def on_trial_done(trial_item):
            trial_item.done_at = time.time()

            if trial_item.reward != 0 and not math.isnan(trial_item.reward):  # success
                improved = hyper_model.history.append(trial_item)
                for callback in hyper_model.callbacks:
                    callback.on_trial_end(hyper_model, trial_item.space_sample,
                                          trial_item.trial_no, trial_item.reward,
                                          improved, trial_item.elapsed)
                success_counter()
            else:
                for callback in hyper_model.callbacks:
                    callback.on_trial_error(hyper_model, trial_item.space_sample, trial_item.trial_no)
                failed_counter()

            if logger.is_info_enabled():
                elapsed = '%.3f' % (trial_item.done_at - trial_item.start_at)
                msg = f'Trial {trial_item.trial_no} done with reward={trial_item.reward}, ' \
                      f'elapsed {elapsed} seconds\n'
                logger.info(msg)
            if trial_store is not None:
                trial_store.put(dataset_id, trial_item)

        pool = DaskExecutorPool(worker_count, queue_size,
                                on_trial_start, on_trial_done,
                                hyper_model._run_trial,
                                X, y, X_val, y_val, fit_kwargs)
        pool.start()

        trial_no = 1
        retry_counter = 0

        while trial_no <= max_trials and pool.running:
            if pool.qsize >= queue_size:
                time.sleep(0.1)
                continue

            space_sample = hyper_model.searcher.sample()
            if hyper_model.history.is_existed(space_sample):
                if retry_counter >= retry_limit:
                    logger.info(f'Unable to take valid sample and exceed the retry limit 1000.')
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

                model_file = '%s/%05d_%s.pkl' % (self.models_dir, trial_no, space_sample.space_id)

                item = DaskTrialItem(space_sample, trial_no, model_file=model_file)
                pool.push(item)

                if logger.is_info_enabled():
                    logger.info(f'Found trial {trial_no}, queue size: {pool.qsize}')
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
                msg = f'{">" * 20} Search trial {trial_no} failed! {"<" * 20}\n' \
                      + f'{e.__class__.__name__}: {e}\n' \
                      + traceback.format_exc() \
                      + '*' * 50
                logger.error(msg)
            finally:
                trial_no += 1
                retry_counter = 0

        # wait trials
        if pool.running:
            logger.info('Search done, wait trial tasks.')
        pool.push(None)  # mark end
        pool.join()

        if logger.is_info_enabled():
            logger.info(f'Search and all trials done, {success_counter.value} success, '
                        f'{failed_counter.value} failed.')

        return trial_no
