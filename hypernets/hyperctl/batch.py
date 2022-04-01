# -*- encoding: utf-8 -*-
import os
from pathlib import Path

import psutil
from typing import Dict, Optional
from hypernets.hyperctl import consts
from hypernets import __version__ as hyn_version


class ExecutionConf:
    def __init__(self, command, data_dir, working_dir):
        self.command = command
        self.data_dir = data_dir
        self.working_dir = working_dir

    def to_config(self):
        return {
            "command": self.command,
            "data_dir": self.data_dir,
            "working_dir": self.working_dir
        }


class ShellJob:
    STATUS_INIT = 'init'
    STATUS_RUNNING = 'running'
    STATUS_SUCCEED = 'succeed'
    STATUS_FAILED = 'failed'

    FINAL_STATUS = [STATUS_SUCCEED, STATUS_FAILED]

    def __init__(self, name, params, resource, execution: ExecutionConf, batch):
        self.name = name
        self.params = params
        self.resource = resource
        self.execution = execution
        self.batch = batch

    @property
    def batch_data_dir(self):
        return self.batch.data_dir_path()

    @property
    def job_data_dir(self):
        return Path(self.execution.data_dir).as_posix()

    @property
    def run_file_path(self):
        return (Path(self.job_data_dir) / "run.sh").as_posix()

    def status_file_path(self, status):
        return (Path(self.batch_data_dir) / f"{self.name}.{status}").as_posix()

    def final_status_files(self):
        return self._status_files([self.STATUS_FAILED, self.STATUS_SUCCEED])

    def status_files(self):
        return self._status_files([self.STATUS_FAILED, self.STATUS_SUCCEED, self.STATUS_RUNNING])

    def _status_files(self, statuses):
        return {status: f"{self.name}.{status}" for status in statuses}

    @property
    def status(self):
        exists_statuses = []
        for status_value, status_file in self.status_files().items():
            abs_status_file = os.path.join(self.batch_data_dir, status_file)
            if os.path.exists(abs_status_file):
                exists_statuses.append((status_value, status_file))

        status_len = len(exists_statuses)
        if status_len > 1:
            files_msg = ",".join(map(lambda _: _[1], exists_statuses))
            raise ValueError(f"Invalid status, multiple status files exists: {files_msg}")
        elif status_len == 1:
            return exists_statuses[0][0]
        else:  # no status file
            return self.STATUS_INIT

    def to_dict(self):
        ret_dict = self.__dict__.copy()
        ret_dict['status'] = self.status
        ret_dict['execution'] = self.execution.__dict__.copy()
        del ret_dict['batch']
        return ret_dict

    def to_config(self):
        return {
            "name": self.name,
            "params": self.params,
            "resource": self.resource,
            "execution": self.execution.to_config()
        }


class ServerConf:  # API server conf
    def __init__(self, host="localhost", port=8060, exit_on_finish=False):
        self.host = host
        self.port = port
        self.exit_on_finish = exit_on_finish

    @property
    def portal(self):
        return f"http://{self.host}:{self.port}"

    def to_config(self):
        return {
            "host": self.host,
            "port": self.port,
            "exit_on_finish": self.exit_on_finish
        }


class BackendConf:
    def __init__(self, type = 'local', conf: Dict = None):
        self.type = type
        if conf is None:
            conf = {}
        self.conf = conf

    def to_config(self):
        return {
            "type": self.type,
            "conf": self.conf
        }


class Batch:

    FILE_SPEC = "spec.json"
    FILE_PID = "server.pid"

    STATUS_NOT_START = "NOT_START"
    STATUS_RUNNING = "RUNNING"
    STATUS_FINISHED = "FINISHED"

    def __init__(self, name, batches_data_dir, backend_conf: BackendConf, server_conf: ServerConf):
        self.name = name
        self.batches_data_dir = batches_data_dir

        self.server_conf = server_conf
        self.backend_conf = backend_conf

        #
        self.jobs = []

    def add_job(self, **kwargs):
        kwargs['batch'] = self
        if kwargs.get('resource') is None:
            kwargs['resource'] = {
                'cpu': -1,
                'mem': -1
            }
        self.jobs.append(ShellJob(**kwargs))

    def status(self):
        pid = self.pid()
        if pid is None:
            return self.STATUS_NOT_START

        try:
            psutil.Process(pid)
            return self.STATUS_RUNNING
        except Exception as e:
            return self.STATUS_FINISHED
        finally:
            pass

    def is_finished(self):
        exists_status = set([job.status for job in self.jobs])
        return exists_status.issubset(set(ShellJob.FINAL_STATUS))

    def spec_file_path(self):
        return self.data_dir_path() / self.FILE_SPEC

    def pid_file_path(self):
        return self.data_dir_path() / self.FILE_PID

    def pid(self):
        pid_file_path = self.pid_file_path()
        if pid_file_path.exists():
            with open(pid_file_path, 'r') as f:
                return int(f.read())
        else:
            return None

    def data_dir_path(self):
        return Path(self.batches_data_dir) / self.name

    def _filter_jobs(self, status):
        return list(filter(lambda j: j.status == status, self.jobs))

    def get_job_by_name(self, job_name) -> Optional[ShellJob]:
        for job in self.jobs:
            if job.name == job_name:
                return job
        return None

    def summary(self):
        def cnt(status):
            return len(self._filter_jobs(status))
        return {
            "name": self.name,
            'status': self.status(),
            'total': len(self.jobs),
            'portal': self.server_conf.portal,
            ShellJob.STATUS_FAILED: cnt(ShellJob.STATUS_FAILED),
            ShellJob.STATUS_INIT: cnt(ShellJob.STATUS_INIT),
            ShellJob.STATUS_SUCCEED: cnt(ShellJob.STATUS_SUCCEED),
            ShellJob.STATUS_RUNNING: cnt(ShellJob.STATUS_RUNNING),
        }

    def to_config(self):
        jobs_config = []
        for job in self.jobs:
            jobs_config.append(job.to_config())

        return {
            "jobs": jobs_config,
            "backend": self.backend_conf.to_config(),
            "name": self.name,
            "server": self.server_conf.to_config(),
            "version": hyn_version
        }


def load_batch(batch_spec_dict, batches_data_dir):
    batch_name = batch_spec_dict['name']
    jobs_dict = batch_spec_dict['jobs']

    server_config_dict = batch_spec_dict.get('server', {})
    backend_config_dict = batch_spec_dict.get('backend', {})

    batch = Batch(batch_name, batches_data_dir, BackendConf(**backend_config_dict), ServerConf(**server_config_dict))
    for job_dict in jobs_dict:
        execution_conf = job_dict.get('execution', {})
        if execution_conf.get('data_dir') is None:
            execution_conf['data_dir'] = (batch.data_dir_path() / job_dict['name']).as_posix()
        if execution_conf.get('working_dir') is None:
            execution_conf['working_dir'] = execution_conf['data_dir']
        job_dict['execution'] = ExecutionConf(**execution_conf)
        batch.add_job(**job_dict)
    return batch


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
