# -*- encoding: utf-8 -*-
# -*- encoding: utf-8 -*-
import os
from pathlib import Path
from typing import Dict, Optional

import psutil

from hypernets.utils import logging

logging.set_level('DEBUG')

logger = logging.getLogger(__name__)


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

    def __init__(self, name, params, command, output_dir, working_dir=None, assets=None, resource=None):
        self.name = name
        self.params = params
        self.resource = resource
        self.command = command
        self.output_dir = output_dir

        if working_dir is None:
            self.working_dir = output_dir
        else:
            self.working_dir = working_dir

        self.assets = assets

        self.start_time = None
        self.end_time = None
        self._status = ShellJob.STATUS_INIT

    def set_status(self, status):
        self._status = status

    @property
    def status(self):
        return self._status

    @property
    def job_data_dir(self):
        return Path(self.output_dir).as_posix()

    @property
    def run_file_path(self):
        return (Path(self.job_data_dir) / "run.sh").as_posix()

    @property
    def resources_path(self):
        return Path(self.job_data_dir) / "resources"  # resources should be copied to working dir

    def final_status_files(self):
        return self._status_files([self.STATUS_FAILED, self.STATUS_SUCCEED])

    def status_files(self):
        return self._status_files([self.STATUS_FAILED, self.STATUS_SUCCEED, self.STATUS_RUNNING])

    def _status_files(self, statuses):
        return {status: f"{self.name}.{status}" for status in statuses}

    def to_dict(self):
        import copy
        config_dict = copy.copy(self.to_config())
        config_dict['status'] = self.status
        return config_dict

    def to_config(self):
        return {
            "name": self.name,
            "params": self.params,
            "resource": self.resource,
            "command": self.command,
            "output_dir": self.output_dir,
            "working_dir": self.working_dir,
            "assets": self.assets
        }

    @property
    def elapsed(self):
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return None


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

    FILE_CONFIG = "config.json"
    FILE_PID = "server.pid"

    STATUS_NOT_START = "NOT_START"
    STATUS_RUNNING = "RUNNING"
    STATUS_FINISHED = "FINISHED"

    def __init__(self, name, batches_data_dir: str):
        self.name = name
        self.batches_data_dir = Path(batches_data_dir)

        self.jobs = []

        self.start_time = None
        self.end_time = None

    def job_status_file_path(self, job_name, status):
        return (self.batch_data_dir_path / f"{job_name}.{status}").as_posix()

    def load_job_status(self, job_name):
        pass

    @property
    def batch_data_dir_path(self):
        return Path(self.batches_data_dir) / self.name

    def add_job(self, **kwargs):
        if kwargs.get('resource') is None:
            kwargs['resource'] = {
                'cpu': -1,
                'mem': -1
            }
            kwargs['assets'] = None
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

    def config_file_path(self):
        return self.data_dir_path() / self.FILE_CONFIG

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

    def get_job_by_name(self, job_name) -> Optional[ShellJob]:
        for job in self.jobs:
            if job.name == job_name:
                return job
        return None

    @property
    def elapsed(self):
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return None

    def summary(self):
        batch = self

        def _filter_jobs(status):
            return list(filter(lambda j: j.status == status, batch.jobs))

        def cnt(status):
            return len(_filter_jobs(status))

        return {
            "name": batch.name,
            'status': batch.status(),
            'total': len(batch.jobs),
            ShellJob.STATUS_FAILED: cnt(ShellJob.STATUS_FAILED),
            ShellJob.STATUS_INIT: cnt(ShellJob.STATUS_INIT),
            ShellJob.STATUS_SUCCEED: cnt(ShellJob.STATUS_SUCCEED),
            ShellJob.STATUS_RUNNING: cnt(ShellJob.STATUS_RUNNING),
        }

