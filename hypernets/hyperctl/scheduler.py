# -*- encoding: utf-8 -*-
import json
import os
import time

from tornado import ioloop
from tornado.ioloop import PeriodicCallback

from hypernets.hyperctl.batch import ShellJob
from hypernets.hyperctl.callbacks import BatchCallback
from hypernets.hyperctl.executor import NoResourceException, ShellExecutor, ExecutorManager
from hypernets.utils import logging as hyn_logging

logger = hyn_logging.getLogger(__name__)


class JobScheduler:
    """a FIFO scheduler"""

    def __init__(self, batch, exit_on_finish, interval, executor_manager: ExecutorManager, callbacks=None):
        self.batch = batch
        self.exit_on_finish = exit_on_finish
        self.executor_manager = executor_manager
        self.callbacks = callbacks if callbacks is not None else []

        self._io_loop_instance = None

        self._timer = PeriodicCallback(self.attempt_scheduling, interval)

    @property
    def interval(self):
        return self._timer.callback_time

    def start(self):
        self.executor_manager.prepare()
        self._timer.start()
        self.batch.start_time = time.time()

        for callback in self.callbacks:
            callback.on_start(self.batch)

        # run in io loop
        self._io_loop_instance = ioloop.IOLoop.instance()
        self._io_loop_instance.start()

    def stop(self):
        if self._io_loop_instance is not None:
            self._io_loop_instance.stop()
        else:
            raise RuntimeError("Not started yet")

    def kill_job(self, job_name):
        # checkout job
        job: ShellJob = self.batch.get_job_by_name(job_name)
        if job is None:
            raise ValueError(f'job {job_name} does not exists ')

        logger.info(f"trying kill job {job_name}, it's status is {job.status} ")

        # check job status
        if job.status != job.STATUS_RUNNING:
            raise RuntimeError(f"job {job_name} in not in {job.STATUS_RUNNING} status but is {job.status} ")

        # find executor and kill
        em = self.executor_manager
        executor = em.get_executor(job)
        logger.info(f"find executor {executor} of job {job_name}")
        if executor is not None:
            em.kill_executor(executor)
            logger.info(f"write failed status file for {job_name}")
            self.change_job_status(job, job.STATUS_FAILED)
        else:
            raise ValueError(f"no executor found for job {job.name}")

    @staticmethod
    def change_job_status(job: ShellJob, next_status):
        current_status = job.status
        target_status_file = job.status_file_path(next_status)
        if next_status == job.STATUS_INIT:
            raise ValueError(f"can not change to {next_status} ")

        elif next_status == job.STATUS_RUNNING:
            if current_status != job.STATUS_INIT:
                raise ValueError(f"only job in {job.STATUS_INIT} can change to {next_status}")

        elif next_status in job.FINAL_STATUS:
            if current_status != job.STATUS_RUNNING:
                raise ValueError(f"only job in {job.STATUS_RUNNING} can change to "
                                 f"{next_status} but now is {current_status}")
            # delete running status file
            running_status_file = job.status_file_path(job.STATUS_RUNNING)
            os.remove(running_status_file)
        else:
            raise ValueError(f"unknown status {next_status}")

        with open(target_status_file, 'w') as f:
            pass

    def _release_executors(self, executor_manager):
        finished = []
        for executor in executor_manager.waiting_executors():
            executor: ShellExecutor = executor
            if executor.status() in ShellJob.FINAL_STATUS:
                finished.append(executor)

        for finished_executor in finished:
            executor_status = finished_executor.status()
            job = finished_executor.job
            logger.info(f"job {job.name} finished with status {executor_status}")
            JobScheduler.change_job_status(job, finished_executor.status())
            job.end_time = time.time()  # update end time
            executor_manager.release_executor(finished_executor)
            self._handle_job_finished(job, finished_executor, job.elapsed)

    def _handle_job_start(self, job, executor):
        for callback in self.callbacks:
            callback: BatchCallback = callback
            callback.on_job_start(self.batch, job, executor)

    def _handle_job_finished(self, job, executor, elapsed):
        for callback in self.callbacks:
            callback: BatchCallback = callback
            callback.on_job_finish(self.batch, job, executor, elapsed)

    def _run_jobs(self, executor_manager, jobs):
        for job in jobs:
            if job.status != job.STATUS_INIT:
                # logger.debug(f"job '{job.name}' status is {job.status}, skip run")
                continue
            try:
                logger.debug(f'trying to alloc resource for job {job.name}')
                executor = executor_manager.alloc_executor(job)
                job.start_time = time.time()  # update start time
                self._handle_job_start(job, executor)
                process_msg = f"{len(executor_manager.allocated_executors())}/{len(jobs)}"
                logger.info(f'allocated resource for job {job.name}({process_msg}), data dir at {job.job_data_dir}')
                # os.makedirs(job.job_data_dir, exist_ok=True)
                JobScheduler.change_job_status(job, job.STATUS_RUNNING)
                executor.run()
            except NoResourceException:
                logger.debug(f"no enough resource for job {job.name} , wait for resource to continue ...")
                break
            except Exception as e:
                JobScheduler.change_job_status(job, job.STATUS_FAILED)  # TODO on job break
                logger.exception(f"failed to run job '{job.name}' ", e)
                continue
            finally:
                pass

    def _handle_on_finished(self):
        for callback in self.callbacks:
            callback: BatchCallback = callback
            callback.on_finish(self.batch, self.batch.elapsed)

    def attempt_scheduling(self):  # attempt_scheduling
        jobs = self.batch.jobs

        # check all jobs finished
        job_finished = self.batch.is_finished()
        if job_finished:
            self.batch.end_time = time.time()
            batch_summary = json.dumps(self.batch.summary())
            logger.info("all jobs finished, stop scheduler:\n" + batch_summary)
            self._timer.stop()  # stop the timer
            if self.exit_on_finish:
                logger.info("exited ioloop")
                self.stop()
            self._handle_on_finished()
            return

        self._release_executors(self.executor_manager)
        self._run_jobs(self.executor_manager, jobs)
