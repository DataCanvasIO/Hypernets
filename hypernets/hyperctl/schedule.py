# -*- encoding: utf-8 -*-
import json
import os
from pathlib import Path

from tornado import ioloop
from tornado.ioloop import PeriodicCallback

from hypernets.hyperctl.batch import Batch, load_batch
from hypernets.hyperctl.batch import ShellJob
from hypernets.hyperctl.executor import NoResourceException, ShellExecutor
from hypernets.utils import logging as hyn_logging, common as common_util

logger = hyn_logging.getLogger(__name__)


class Scheduler:

    def __init__(self, batch, exit_on_finish, interval, executor_manager):
        self.batch = batch
        self.exit_on_finish = exit_on_finish
        self.executor_manager = executor_manager
        self._timer = PeriodicCallback(self.schedule, interval)

    def start(self):
        self._timer.start()

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
            Scheduler.change_job_status(job, finished_executor.status())
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
                Scheduler.change_job_status(job, job.STATUS_RUNNING)
                executor.run()
            except NoResourceException:
                logger.debug(f"no enough resource for job {job.name} , wait for resource to continue ...")
                break
            except Exception as e:
                Scheduler.change_job_status(job, job.STATUS_FAILED)
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
                logger.info("stop ioloop")
                ioloop.IOLoop.instance().stop()
            return

        self._check_executors(self.executor_manager)
        self._dispatch_jobs(self.executor_manager, jobs)


def _start_api_server(batch: Batch):
    # create web app
    logger.info(f"start daemon server at: {batch.daemon_conf.portal}")
    from hypernets.hyperctl.server import create_batch_manage_webapp
    create_batch_manage_webapp().listen(batch.daemon_conf.port)

    # run io loop
    ioloop.IOLoop.instance().start()


def run_batch(batch: Batch, batches_data_dir):
    prepare_batch(batch, batches_data_dir)

    _start_api_server(batch)


def prepare_batch(batch: Batch, batches_data_dir):

    batches_data_dir = Path(batches_data_dir)
    logger.info(f"batches_data_path: {batches_data_dir.absolute()}")
    logger.info(f"batch name: {batch.name}")

    # check jobs status
    for job in batch.jobs:
        if job.status != job.STATUS_INIT:
            if job.status == job.STATUS_RUNNING:
                logger.warning(f"job '{job.name}' status is {job.status} in the begining,"
                               f"it may have run and will not run again this time, "
                               f"you can remove it's status file and working dir to retry the job")
            else:
                logger.info(f"job '{job.name}' status is {job.status} means it's finished, skip to run ")
            continue

    # prepare batch data dir
    if batch.data_dir_path().exists():
        logger.info(f"batch {batch.name} already exists, run again")
    else:
        os.makedirs(batch.data_dir_path(), exist_ok=True)

    # write batch config
    batch_spec_file_path = batch.spec_file_path()

    with open(batch_spec_file_path, 'w', newline='\n') as f:
        json.dump(batch.to_config(), f, indent=4)

    # create executor manager
    from hypernets.hyperctl.executor import create_executor_manager
    executor_manager = create_executor_manager(batch.backend_conf, batch.daemon_conf)

    # start scheduler
    Scheduler(batch, batch.daemon_conf.exit_on_finish, 5000, executor_manager).start()

    # write pid file
    with open(batch.pid_file_path(), 'w', newline='\n') as f:
        f.write(str(os.getpid()))


def run_batch_config(config_dict, batches_data_dir):
    # add batch name
    if config_dict.get('name') is None:
        batch_name = common_util.generate_short_id()
        logger.debug(f"generated batch name {batch_name}")
        config_dict['name'] = batch_name

    # add job name
    jobs_dict = config_dict['jobs']
    for job_dict in jobs_dict:
        if job_dict.get('name') is None:
            job_name = common_util.generate_short_id()
            logger.debug(f"generated job name {job_name}")
            job_dict['name'] = job_name

    batches_data_dir = Path(batches_data_dir)

    batch = load_batch(config_dict, batches_data_dir)
    prepare_batch(batch, batches_data_dir)

    _start_api_server(batch)
    # TODO: check return
    return batch
