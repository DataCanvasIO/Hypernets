# -*- encoding: utf-8 -*-
import argparse
import codecs
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Optional, Awaitable

import prettytable as pt
import yaml
from tornado import ioloop
from tornado.ioloop import PeriodicCallback
from tornado.log import app_log
from tornado.web import RequestHandler, Finish, HTTPError, Application

from hypernets import __version__ as current_version
from hypernets.hyperctl import Context, set_context
from hypernets.hyperctl import consts
from hypernets.hyperctl import dao
from hypernets.hyperctl import api
from hypernets.hyperctl.batch import Batch, DaemonConf, BackendConf
from hypernets.hyperctl.batch import ShellJob
from hypernets.hyperctl.dao import change_job_status
from hypernets.hyperctl.executor import RemoteSSHExecutorManager, NoResourceException, SSHRemoteMachine, \
    LocalExecutorManager, ShellExecutor
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
            change_job_status(job, finished_executor.status())
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
                change_job_status(job, job.STATUS_RUNNING)
                executor.run()
            except NoResourceException:
                logger.debug(f"no enough resource for job {job.name} , wait for resource to continue ...")
                break
            except Exception as e:
                change_job_status(job, job.STATUS_FAILED)
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


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return yaml.load(content, Loader=yaml.CLoader)


def load_json(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return json.loads(content)


def _copy_item(src, dest, key):
    v = src.get(key)
    if v is not None:
        dest[key] = v


def run_generate_job_specs(template, output):
    yaml_file = template
    output_path = Path(output)
    # 1. validation
    # 1.1. checkout output
    if output_path.exists():
        raise FileExistsError(output)

    # load file
    config_dict = load_yaml(yaml_file)

    # 1.3. check values should be array
    assert "params" in config_dict
    params = config_dict['params']
    for k, v in params.items():
        if not isinstance(v, list):
            raise ValueError(f"Value of param '{k}' should be list")

    # 1.4. check command exists
    assert "execution" in config_dict
    assert 'command' in config_dict['execution']

    # 2. combine params to generate jobs
    job_param_names = params.keys()
    param_values = [params[_] for _ in job_param_names]

    def make_job_dict(job_param_values):
        job_params_dict = dict(zip(job_param_names, job_param_values))
        job_dict = {
            "name": common_util.generate_short_id(),
            "params": job_params_dict,
            "execution": config_dict['execution']
        }
        _copy_item(config_dict, job_dict, 'resource')
        return job_dict

    jobs = [make_job_dict(_) for _ in itertools.product(*param_values)]

    # 3. merge to bath spec
    batch_spec = {
        "jobs": jobs,
        'name': config_dict.get('name', common_util.generate_short_id()),
        "version": config_dict.get('version', current_version)
    }

    _copy_item(config_dict, batch_spec, 'backend')
    _copy_item(config_dict, batch_spec, 'daemon')

    # 4. write to file
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w', newline='\n') as f:
        f.write(json.dumps(batch_spec, indent=4))
    return batch_spec


def run_batch_from_config_file(config, batches_data_dir=None):
    config_dict = load_json(config)
    prepare_batch(config_dict, batches_data_dir)


def load_batch(batch_spec_dict, batches_data_dir):
    batch_name = batch_spec_dict['name']
    jobs_dict = batch_spec_dict['jobs']

    default_daemon_conf = consts.default_daemon_conf()

    user_daemon_conf = batch_spec_dict.get('daemon')
    if user_daemon_conf is not None:
        default_daemon_conf.update(user_daemon_conf)

    backend_config = batch_spec_dict.get('backend', {})

    batch = Batch(batch_name, batches_data_dir, BackendConf(**backend_config), DaemonConf(**default_daemon_conf))
    for job_dict in jobs_dict:
        batch.add_job(**job_dict)
    return batch


def load_batch_data_dir(batches_data_dir, batch_name):
    spec_file_path = Path(batches_data_dir) / batch_name / Batch.FILE_SPEC
    if not spec_file_path.exists():
        raise RuntimeError(f"batch {batch_name} not exists ")
    batch_spec_dict = load_json(spec_file_path)
    return load_batch(batch_spec_dict, batches_data_dir)


def _run_batch(batch: Batch, batches_data_dir=None):
    prepare_batch(batch, batches_data_dir)


    # create web app
    logger.info(f"start daemon server at: {batch.daemon_conf.portal}")
    from hypernets.hyperctl.server import create_batch_manage_webapp
    create_batch_manage_webapp().listen(batch.daemon_conf.port)

    # run io loop
    ioloop.IOLoop.instance().start()


def prepare_batch(batch: Batch, batches_data_dir=None):

    logger.info(f"batches_data_path: {batches_data_dir.absolute()}")
    logger.info(f"batch name: {batch.name}")

    # check remote host setting
    daemon_host = batch.daemon_conf.host
    if consts.HOST_LOCALHOST == daemon_host:
        logger.warning("recommended that set IP address that can be accessed in remote machines, "
                       "but now it's \"localhost\", and the task executed on the remote machines "
                       "may fail because it can't get information from the daemon server,"
                       " you can set it in `daemon.host` ")

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
    executor_manager = create_executor_manager(batch.backend_conf, batch.daemon_conf.portal)
    # start scheduler
    # Scheduler(batch, batch.daemon_conf.exit_on_finish, 5000, ).start()
    Scheduler(batch, batch.daemon_conf.exit_on_finish, 5000, executor_manager).start()
    # set context
    c = Context(executor_manager, batch)
    set_context(c)

    # write pid file
    with open(batch.pid_file_path(), 'w', newline='\n') as f:
        f.write(str(os.getpid()))


def run_batch_batch_config(config_dict, batches_data_dir=None):
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

    batches_data_dir = get_batches_data_dir(batches_data_dir)
    batches_data_dir = Path(batches_data_dir)

    batch = load_batch(config_dict, batches_data_dir)
    prepare_batch(batch, batches_data_dir)


def get_batches_data_dir(batches_data_dir):
    if batches_data_dir is None:
        bdd_env = os.environ.get(consts.KEY_ENV_BATCHES_DATA_DIR)
        if bdd_env is None:
            bdd_default = Path("~/hyperctl-batches-data-dir").expanduser().as_posix()
            logger.debug(f"use default batches_data_dir path: {bdd_default}")
            return bdd_default
        else:
            logger.debug(f"found batches_data_dir setting in environment: {bdd_env}")
            return bdd_env
    else:
        return batches_data_dir
