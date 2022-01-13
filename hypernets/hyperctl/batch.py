import os
from pathlib import Path

import psutil

from hypernets.utils import common as common_util


class ExecutionConf:
    def __init__(self, command, data_dir, working_dir):
        self.command = command
        self.data_dir = data_dir
        self.working_dir = working_dir


class ShellJob:
    STATUS_INIT = 'init'
    STATUS_RUNNING = 'running'
    STATUS_SUCCEED = 'succeed'
    STATUS_FAILED = 'failed'

    FINAL_STATUS = [STATUS_SUCCEED, STATUS_FAILED]

    def __init__(self, name, params, resource, execution, batch):
        self.name = name
        self.params = params
        self.resource = resource

        if execution is None:
            execution = {}
        if execution.get('data_dir') is None:
            execution['data_dir'] = (batch.data_dir_path() / self.name).as_posix()
        if execution.get('working_dir') is None:
            execution['working_dir'] = execution['data_dir']
        self.execution = ExecutionConf(**execution)

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


class DaemonConf:
    def __init__(self, host, port, exit_on_finish=False):
        self.host = host
        self.port = port
        self.exit_on_finish = exit_on_finish

    @property
    def portal(self):
        return f"http://{self.host}:{self.port}"


class Batch:

    FILE_SPEC = "spec.json"
    FILE_PID = "daemon.pid"

    STATUS_NOT_START = "NOT_START"
    STATUS_RUNNING = "RUNNING"
    STATUS_FINISHED = "FINISHED"

    def __init__(self, name, batches_data_dir, daemon_conf: DaemonConf):
        self.name = name
        self.batches_data_dir = batches_data_dir

        self.daemon_conf = daemon_conf

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

    def summary(self):
        def cnt(status):
            return len(self._filter_jobs(status))
        return {
            "name": self.name,
            'status': self.status(),
            'total': len(self.jobs),
            'portal': self.daemon_conf.portal,
            ShellJob.STATUS_FAILED: cnt(ShellJob.STATUS_FAILED),
            ShellJob.STATUS_INIT: cnt(ShellJob.STATUS_INIT),
            ShellJob.STATUS_SUCCEED: cnt(ShellJob.STATUS_SUCCEED),
            ShellJob.STATUS_RUNNING: cnt(ShellJob.STATUS_RUNNING),
        }
