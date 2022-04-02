# -*- encoding: utf-8 -*-
import json
import os
from pathlib import Path

from tornado import ioloop
from tornado.ioloop import PeriodicCallback

from hypernets.hyperctl.batch import Batch
from hypernets.hyperctl.batch import ShellJob
from hypernets.hyperctl.executor import NoResourceException, ShellExecutor, ExecutorManager
from hypernets.hyperctl.utils import load_json, http_portal
from hypernets.utils import logging as hyn_logging, common as common_util
from hypernets import __version__ as hyn_version

logger = hyn_logging.getLogger(__name__)


class JobScheduler:

    def __init__(self, batch, exit_on_finish, interval, executor_manager: ExecutorManager):
        self.batch = batch
        self.exit_on_finish = exit_on_finish
        self.executor_manager = executor_manager
        self._timer = PeriodicCallback(self.schedule, interval)

    @property
    def interval(self):
        return self._timer.callback_time

    def start(self):
        self.executor_manager.prepare()
        self._timer.start()

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

    @staticmethod
    def _check_executors(executor_manager):
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
            executor_manager.release_executor(finished_executor)

    @staticmethod
    def _dispatch_jobs(executor_manager, jobs):
        for job in jobs:
            if job.status != job.STATUS_INIT:
                # logger.debug(f"job '{job.name}' status is {job.status}, skip run")
                continue
            try:
                logger.debug(f'trying to alloc resource for job {job.name}')
                executor = executor_manager.alloc_executor(job)
                process_msg = f"{len(executor_manager.allocated_executors())}/{len(jobs)}"
                logger.info(f'allocated resource for job {job.name}({process_msg}), data dir at {job.job_data_dir} ')
                # os.makedirs(job.job_data_dir, exist_ok=True)
                JobScheduler.change_job_status(job, job.STATUS_RUNNING)
                executor.run()
            except NoResourceException:
                logger.debug(f"no enough resource for job {job.name} , wait for resource to continue ...")
                break
            except Exception as e:
                JobScheduler.change_job_status(job, job.STATUS_FAILED)
                logger.exception(f"failed to run job '{job.name}' ", e)
                continue
            finally:
                pass

    def schedule(self):
        jobs = self.batch.jobs
        # check all jobs finished
        job_finished = self.batch.is_finished()
        if job_finished:
            batch_summary = json.dumps(self.batch.summary())
            logger.info("all jobs finished, stop scheduler:\n" + batch_summary)
            self._timer.stop()  # stop the timer
            if self.exit_on_finish:
                logger.info("exited ioloop")
                ioloop.IOLoop.instance().stop()
            return

        self._check_executors(self.executor_manager)
        self._dispatch_jobs(self.executor_manager, jobs)
