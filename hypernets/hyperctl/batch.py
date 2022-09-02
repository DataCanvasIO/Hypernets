# -*- encoding: utf-8 -*-
import json
import tempfile
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import psutil

from hypernets.hyperctl import consts
from hypernets.utils import logging
from hypernets.utils.common import generate_short_id

logger = logging.getLogger(__name__)


class _ShellJob:  # internal class

    STATUS_INIT = 'init'
    STATUS_RUNNING = 'running'
    STATUS_SUCCEED = 'succeed'
    STATUS_FAILED = 'failed'

    FINAL_STATUS = [STATUS_SUCCEED, STATUS_FAILED]

    def __init__(self,  *, name, batch, params, working_dir=None, assets=None, resource=None):

        self.name = name
        self.batch = batch
        self.params = params
        self.resource = resource

        # write job files to tmp
        self.data_dir_path = Path(tempfile.gettempdir()) \
                             / f"{consts.JOB_DATA_DIR_PREFIX}{self.batch.name}_{self.name}_{generate_short_id()}"

        if working_dir is None:
            self.working_dir = self.data_dir_path.as_posix()
        else:
            self.working_dir = working_dir

        self.assets = [] if assets is None else assets

        self.start_datetime = None
        self.end_datetime = None

        self._status = _ShellJob.STATUS_INIT
        self._ext = {}

    @property
    def ext(self):
        return self._ext

    def set_ext(self, ext):
        self._ext = ext

    def state_data_file(self):  # on master node
        return (self.batch.data_dir_path / f"{self.name}.json").as_posix()

    def state_data(self):
        with open(self.state_data_file(), 'r') as f:
            return json.load(f)

    def set_status(self, status):
        self._status = status

    @property
    def status(self):
        return self._status

    @property
    def run_file(self):
        return (self.data_dir_path / "run.sh").as_posix()

    @property
    def resources_path(self):
        return self.data_dir_path / "resources"  # resources should be copied to working dir

    def to_dict(self):
        import copy
        config_dict = copy.copy(self.to_config())
        return config_dict

    def to_config(self):
        return {
            "name": self.name,
            "params": self.params,
            "resource": self.resource,
            "working_dir": self.working_dir,
            "assets": self.assets
        }

    @property
    def elapsed(self):
        if self.start_datetime is not None and self.end_datetime is not None:
            return self.end_datetime - self.start_datetime
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

    def __init__(self, *, name, job_command, data_dir: str):
        self.name = name
        self.job_command = job_command
        self.data_dir = data_dir

        self._jobs_dict = OrderedDict()
        self.start_time = None
        self.end_time = None

    @property
    def data_dir_path(self):
        return Path(self.data_dir)

    @property
    def jobs(self):
        return list(self._jobs_dict.values())

    def job_status_file_path(self, job_name, status):
        return (self.data_dir_path / f"{job_name}.{status}").as_posix()

    def job_state_data_file_path(self, job_name):
        return (self.data_dir_path / f"{job_name}.json").as_posix()

    def status_files(self):
        return self._status_files([_ShellJob.STATUS_FAILED, _ShellJob.STATUS_SUCCEED, _ShellJob.STATUS_RUNNING])

    def _status_files(self, statuses):
        return {status: f"{self.name}.{status}" for status in statuses}

    def get_persisted_job_status(self, job_name):
        exists_statuses = []
        for status_value, status_file in self.status_files().items():
            abs_status_file = self.job_status_file_path(job_name, status_value)
            if os.path.exists(abs_status_file):
                exists_statuses.append((status_value, status_file))

        status_len = len(exists_statuses)
        if status_len > 1:
            files_msg = ",".join(map(lambda _: _[1], exists_statuses))
            raise ValueError(f"Invalid status, multiple status files exists: {files_msg}")
        elif status_len == 1:
            return exists_statuses[0][0]
        else:  # no status file
            return _ShellJob.STATUS_INIT

    def add_job(self, name, **kwargs):
        assert name not in self._jobs_dict, f'job {name} is already exists'  # check job name
        self._jobs_dict[name] = _ShellJob(name=name, batch=self, **kwargs)

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
        return exists_status.issubset(set(_ShellJob.FINAL_STATUS))

    def config_file_path(self):
        return self.data_dir_path / self.FILE_CONFIG

    def pid_file_path(self):
        return self.data_dir_path / self.FILE_PID

    def pid(self):
        pid_file_path = self.pid_file_path()
        if pid_file_path.exists():
            with open(pid_file_path, 'r') as f:
                return int(f.read())
        else:
            return None

    def get_job_by_name(self, job_name) -> Optional[_ShellJob]:
        for job in self.jobs:
            if job.name == job_name:
                return job
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
            _ShellJob.STATUS_FAILED: cnt(_ShellJob.STATUS_FAILED),
            _ShellJob.STATUS_INIT: cnt(_ShellJob.STATUS_INIT),
            _ShellJob.STATUS_SUCCEED: cnt(_ShellJob.STATUS_SUCCEED),
            _ShellJob.STATUS_RUNNING: cnt(_ShellJob.STATUS_RUNNING),
        }

    @property
    def elapsed(self):
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return None
