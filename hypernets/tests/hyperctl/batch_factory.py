import _thread
import sys
import tempfile
import time
from pathlib import Path

import pytest

from hypernets.hyperctl import api
from hypernets.hyperctl import scheduler, utils
from hypernets.hyperctl.batch import BackendConf, ServerConf, ExecutionConf
from hypernets.hyperctl.batch import ShellJob, Batch
from hypernets.hyperctl.executor import LocalExecutorManager, RemoteSSHExecutorManager
from hypernets.tests.utils import ssh_utils_test
from hypernets.utils import is_os_windows
from hypernets.utils.common import generate_short_id
import os

SRC_DIR = os.path.dirname(__file__)


def create_minimum_batch(command="pwd"):
    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    batch = Batch("minimum-batch", batches_data_dir,
                  backend_conf=BackendConf(type='local'),
                  server_conf=ServerConf('localhost', 8063, exit_on_finish=True))

    data_dir = (Path(batches_data_dir)/ batch.name / "job1").absolute().as_posix()

    execution_conf = ExecutionConf(command=command, data_dir=data_dir, working_dir=data_dir)

    job_params = {"learning_rate": 0.1}
    batch.add_job(name='job1', params=job_params, resource=None, execution=execution_conf)
    return batch


def create_local_batch():
    job1_name = "job1"
    job2_name = "job2"
    daemon_port = 8062

    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    batch = Batch("local-batch", batches_data_dir,
                  backend_conf=BackendConf(type='local'),
                  server_conf=ServerConf('localhost', daemon_port, exit_on_finish=True))

    job1_data_dir = (Path(batches_data_dir)/ job1_name).absolute().as_posix()

    local_example_script = os.path.join(SRC_DIR, "plain_job_script.py")

    batch.add_job(name=job1_name,
                  params={"learning_rate": 0.1},
                  execution=ExecutionConf(command=f"{sys.executable} {local_example_script}",
                                          data_dir=job1_data_dir,
                                          working_dir=job1_data_dir))

    job2_data_dir = (Path(batches_data_dir) / job2_name).absolute().as_posix()
    batch.add_job(name=job2_name,
                  params={"learning_rate": 0.2},
                  execution=ExecutionConf(command=f"{sys.executable} {local_example_script}",
                                          data_dir=job2_data_dir,
                                          working_dir=job2_data_dir))
    return batch
