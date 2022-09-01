# -*- encoding: utf-8 -*-
import json
import os
import time

import tornado
from tornado.ioloop import PeriodicCallback


from hypernets.hyperctl.batch import _ShellJob, Batch
from hypernets.hyperctl.callbacks import BatchCallback
from hypernets.hyperctl.executor import NoResourceException, ShellExecutor, ExecutorManager
from hypernets.utils import logging as hyn_logging

logger = hyn_logging.getLogger(__name__)


class JobScheduler:
    """a FIFO scheduler"""

    def __init__(self, *, batch, exit_on_finish, interval, executor_manager: ExecutorManager,
                 callbacks=None, signal_file=None, independent_tmp=True):

        self.batch = batch
        self.exit_on_finish = exit_on_finish
        self.executor_manager = executor_manager
        self.callbacks = callbacks if callbacks is not None else []
        self.signal_file = signal_file
        self.independent_tmp = independent_tmp

        self._io_loop_instance = None

        self._timer = PeriodicCallback(self.attempt_scheduling, interval)

        self._n_skipped = 0
        self._n_allocated = 0
        self._selected_jobs = []

    @property
    def n_skipped(self):
        return self._n_skipped

    @property
    def n_allocated(self):
        return self._n_allocated

    @property
    def interval(self):
        return self._timer.callback_time

    def start(self):
        self.executor_manager.prepare()
        self._timer.start()
        self.batch.start_datetime = time.time()

        # stats finished jobs
        for job in self.batch.jobs:
            job_status = job.status
            if job_status != _ShellJob.STATUS_INIT:
                if job_status == job.STATUS_RUNNING:
                    logger.warning(f"job '{job.name}' status is {job_status} in the beginning,"
                                   f"it may have run and will not run again this time, "
                                   f"you can remove it's status file: "
                                   f"{self.batch.job_status_file_path(job_name=job.name, status=job_status)} "
                                   f"and data dir(maybe in remote): {job.data_dir_path} to retry the job")
                else:
                    logger.info(f"job '{job.name}' status is {job_status} means it's finished, skip to run ")

                self._n_skipped = self.n_skipped + 1
            else:
                self._selected_jobs.append(job)

        for callback in self.callbacks:
            callback.on_start(self.batch)

        # run in io loop
        self._io_loop_instance = tornado.ioloop.IOLoop.instance()
        logger.info('starting io loop')
        self._io_loop_instance.start()
        logger.info('exited io loop')

    def stop(self):
        if self._io_loop_instance is not None:
            self._io_loop_instance.add_callback(self._io_loop_instance.stop)  # let ioloop stop itself
            # self._io_loop_instance.stop() # This is not work for another Thread to stop the ioloop
            logger.info("add a stop callback to ioloop")
        else:
            raise RuntimeError("Not started yet")

    def kill_job(self, job_name):
        # checkout job
        job: _ShellJob = self.batch.get_job_by_name(job_name)
        if job is None:
            raise ValueError(f'job {job_name} does not exists ')
        job_status = job.status

        logger.info(f"trying kill job {job_name}, it's status is {job_status} ")

        # check job status

        if job_status != job.STATUS_RUNNING:
            raise RuntimeError(f"job {job_name} in not in {job.STATUS_RUNNING} status but is {job_status} ")

        # find executor and kill
        em = self.executor_manager
        executor = em.get_executor(job)
        logger.info(f"find executor {executor} of job {job_name}")
        if executor is not None:
            em.kill_executor(executor)
            logger.info(f"write failed status file for {job_name}")
            self._change_job_status(job, job.STATUS_FAILED)
        else:
            raise ValueError(f"no executor found for job {job.name}")

    def _change_job_status(self, job: _ShellJob, next_status):
        self.change_job_status(self.batch, job, next_status)

    @staticmethod
    def change_job_status(batch: Batch, job: _ShellJob, next_status):
        current_status = job.status
        target_status_file = batch.job_status_file_path(job_name=job.name, status=next_status)

        def touch(f_path):
            with open(f_path, 'w') as f:
                pass

        if next_status == job.STATUS_INIT:
            raise ValueError(f"can not change to {next_status} ")
        elif next_status == job.STATUS_RUNNING:
            if current_status != job.STATUS_INIT:
                raise ValueError(f"only job in {job.STATUS_INIT} can change to {next_status}")

            touch(target_status_file)
            job.set_status(next_status)
            reload_status = batch.get_persisted_job_status(job_name=job.name)
            assert reload_status == next_status, f"change job status failed, current status is {reload_status}," \
                                                 f" expected status is {next_status}"

        elif next_status in job.FINAL_STATUS:
            if current_status != job.STATUS_RUNNING:
                raise ValueError(f"only job in {job.STATUS_RUNNING} can change to "
                                 f"{next_status} but now is {current_status}")
            # remove running status
            os.remove(batch.job_status_file_path(job_name=job.name, status=job.STATUS_RUNNING))

            # job.set_status(next_status)
            touch(target_status_file)
            reload_status = batch.get_persisted_job_status(job_name=job.name)
            assert reload_status == next_status, f"change job status failed, current status is {reload_status}," \
                                                 f" expected status is {next_status}"
            job.set_status(next_status)

        else:
            raise ValueError(f"unknown status {next_status}")

    def _release_executors(self, executor_manager):
        finished = []
        for executor in executor_manager.waiting_executors():
            executor: ShellExecutor = executor
            if executor.status() in _ShellJob.FINAL_STATUS:
                finished.append(executor)

        for finished_executor in finished:
            executor_status = finished_executor.status()
            job = finished_executor.job
            executor_manager.release_executor(finished_executor)  # ensure close connection
            logger.info(f"job {job.name} finished with status {executor_status}")
            self._change_job_status(job, executor_status)
            job.end_datetime = time.time()  # update end time

            if executor_status == _ShellJob.STATUS_SUCCEED:
                self._handle_job_succeed(job, finished_executor, job.elapsed)
            else:
                self._handle_job_failed(job, finished_executor, job.elapsed)

    def _handle_callbacks(self, func):
        for callback in self.callbacks:
            try:
                callback: BatchCallback = callback
                func(callback)
            except Exception as e:
                logger.warning("handle callback failed", e)

    def _handle_job_start(self, job, executor):
        def f(callback):
            callback.on_job_start(self.batch, job, executor)
        self._handle_callbacks(f)

    def _handle_job_succeed(self, job, executor, elapsed):

        # write state data file
        job_state_data = {
            "start_datetime": job.start_datetime,
            "elapsed": elapsed,
            "end_datetime": job.end_datetime,
            'ext':  job.ext
        }

        # check state file
        state_data_path = job.state_data_file()

        if os.path.exists(state_data_path):
            logger.info(f"state data file {state_data_path} already exists will be overwritten ")

        # write state
        with open(state_data_path, 'w') as f:
            json.dump(job_state_data, f)

        # notify callbacks
        def f(callback):
            callback.on_job_succeed(self.batch, job, executor, elapsed)
        self._handle_callbacks(f)

    def _handle_job_broken(self, job, exception):
        def f(callback):
            callback.on_job_break(self.batch, job, exception)

        self._handle_callbacks(f)

    def _handle_job_failed(self, job, executor, elapsed):
        def f(callback):
            callback.on_job_failed(self.batch, job, executor, elapsed)

        self._handle_callbacks(f)

    def _run_jobs(self, executor_manager):
        jobs = self._selected_jobs
        for job in jobs:
            if job.status != job.STATUS_INIT:
                # logger.info(f"job '{job.name}' status is {job.status}, skip run")
                continue
            # logger.debug(f'trying to alloc resource for job {job.name}')
            try:
                executor = executor_manager.alloc_executor(job)
            except NoResourceException:
                # logger.debug(f"no enough resource for job {job.name} , wait for resource to continue ...")
                break
            except Exception as e:
                # skip the job, and do not clean the executor
                self._change_job_status(job, job.STATUS_FAILED)
                self._handle_job_broken(job, e)
                logger.exception(f"failed to alloc resource for job '{job.name}'")
                continue

            self._n_allocated = self.n_allocated + 1
            job.start_datetime = time.time()  # update start time
            self._change_job_status(job, job.STATUS_RUNNING)  # should before callback

            self._handle_job_start(job, executor)
            logger.info(f'ScheduleSummary(allocated={len(executor_manager.allocated_executors())}, total={len(jobs)})')
            try:
                executor.run(independent_tmp=self.independent_tmp)
            except Exception as e:
                logger.exception(f"failed to run job '{job.name}' ", e)
                self._change_job_status(job, job.STATUS_FAILED)
                executor_manager.release_executor(executor)
                self._handle_job_broken(job, e)
                continue
            finally:
                pass

    def _handle_on_finished(self):
        for callback in self.callbacks:
            callback: BatchCallback = callback
            callback.on_finish(self.batch, self.batch.elapsed)

    def attempt_scheduling(self):  # attempt_scheduling

        # check all jobs finished
        batch_finished = self.batch.is_finished()
        if batch_finished:
            self.batch.end_datetime = time.time()
            batch_summary = json.dumps(self.batch.summary())
            logger.info("all jobs finished, stop scheduler:\n" + batch_summary)
            self._timer.stop()  # stop the timer
            if self.exit_on_finish:
                self.stop()
            self._handle_on_finished()
            return

        # check signal file
        if self.signal_file is not None:
            if os.path.exists(self.signal_file):
                logger.info(f"suspend scheduling: {self.signal_file}")
                return

        self._release_executors(self.executor_manager)
        self._run_jobs(self.executor_manager)
