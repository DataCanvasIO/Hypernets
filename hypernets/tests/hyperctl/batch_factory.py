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


def create_minimum_batch():
    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    batch = Batch("minimum-batch", batches_data_dir,
                  backend_conf=BackendConf(type='local'),
                  server_conf=ServerConf('localhost', 8063, exit_on_finish=True))

    data_dir = (Path(batches_data_dir)/"job1").absolute().as_posix()

    execution_conf = ExecutionConf(command="pwd", data_dir=data_dir, working_dir=data_dir)

    job_params = {"learning_rate": 0.1}
    batch.add_job(name='job1', params=job_params, resource=None, execution=execution_conf)
    return batch


def create_local_batch():

    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    batch = Batch("local-batch", batches_data_dir,
                  backend_conf=BackendConf(type='local'),
                  server_conf=ServerConf('localhost', 8063, exit_on_finish=True))

    data_dir = (Path(batches_data_dir)/"job1").absolute().as_posix()

    execution_conf = ExecutionConf(command="pwd", data_dir=data_dir, working_dir=data_dir)

    job_params = {"learning_rate": 0.1}
    batch.add_job(name='job1', params=job_params, resource=None, execution=execution_conf)

    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")

    local_example_script = Path("hypernets/tests/hyperctl/plain_job_script.py").absolute()
    config_dict = {
        "jobs": [
            {
                "name": job1_name,
                "params": {
                    "learning_rate": 0.1
                },
                "resource": {
                    "cpu": 2
                },
                "execution": {
                    "command": f"{sys.executable} {local_example_script}",
                    "working_dir": "/tmp"
                }
            }, {
                "name": job2_name,
                "params": {
                    "learning_rate": 0.1
                },
                "resource": {
                    "cpu": 2
                },
                "execution": {
                    "command": f"{sys.executable} {local_example_script}",
                    "working_dir": "/tmp",
                }
            }
        ],
        "backend": {
            "type": "local",
            "conf": {}
        },
        "name": batch_name,
        "server": {
            "port": 8061,
            "exit_on_finish": True
        }
    }

    batch = scheduler.run_batch_config(config_dict, batches_data_dir)



