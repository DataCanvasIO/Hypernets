import _thread
import sys
import tempfile
import time
from pathlib import Path

import pytest

from hypernets.hyperctl import api
from hypernets.hyperctl import scheduler, utils
from hypernets.hyperctl.batch import BackendConf, ServerConf, ExecutionConf
from hypernets.hyperctl.batch import ShellJob, Batch, load_batch
from hypernets.hyperctl.executor import LocalExecutorManager, RemoteSSHExecutorManager
from hypernets.tests.hyperctl.batch_factory import create_minimum_batch
from hypernets.tests.utils import ssh_utils_test
from hypernets.utils import is_os_windows
from hypernets.utils.common import generate_short_id


def test_batch_to_config():
    # 1. create a batch
    batch = create_minimum_batch()
    # 2. to_config
    batch_config_dict = batch.to_config()

    # 3. asset config content
    # 3.1. check jobs
    jobs_config = batch_config_dict['jobs']
    assert len(jobs_config) == 1
    job_config = jobs_config[0]

    assert job_config['name'] == 'job1'
    assert job_config['params']["learning_rate"] == 0.1
    execution_conf_dict = job_config['execution']
    assert execution_conf_dict['command'] == 'pwd'
    assert execution_conf_dict['data_dir']
    assert execution_conf_dict['working_dir']

    # 3.2 check backend
    backend_config = batch_config_dict['backend']
    assert backend_config['type'] == 'local'

    # 3.3 check server config
    server_config = batch_config_dict['server']
    assert server_config['host'] == 'localhost'
    assert server_config['port'] == 8063
    assert server_config['exit_on_finish'] is True

    # 3.4. check version
    assert batch_config_dict['version']


def test_load_local_batch_from_config():
    # 1. load batch from config
    local_batch = {
        "name": "local_batch_test",
        "jobs": [
            {
                "name": "job1",
                "params": {
                    "learning_rate": 0.1
                },
                "execution": {
                    "command": "pwd"
                }
            }, {
                "name": "job2",
                "params": {
                    "learning_rate": 0.2
                },
                "execution": {
                    "command": "sleep 3",
                    "working_dir": "/tmp",
                    "data_dir": "/tmp/hyperctl-batch-data/job2"
                }
            }
        ],
        "backend": {
            "type": "local",
            "conf": {}
        },
        "server": {
            "host": "local_machine",
            "port": 18060,
            "exit_on_finish": False
        },
        "version": 2.5
    }
    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    batch = load_batch(local_batch, batches_data_dir)

    # 2. assert batch
    assert batch.name == "local_batch_test"
    jobs = batch.jobs
    assert len(jobs) == 2
    job1: ShellJob = jobs[0]
    assert isinstance(job1, ShellJob)
    assert job1.name == "job1"
    assert job1.params['learning_rate'] == 0.1
    assert job1.execution.command == "pwd"
    assert job1.execution.data_dir == (Path(batches_data_dir) / "local_batch_test" / "job1").absolute().as_posix()
    assert job1.execution.working_dir == (Path(batches_data_dir) / "local_batch_test" / "job1").absolute().as_posix()

    job2: ShellJob = jobs[1]
    assert isinstance(job2, ShellJob)
    assert job2.name == "job2"
    assert job2.params['learning_rate'] == 0.2
    assert job2.execution.command == "sleep 3"
    assert job2.execution.data_dir == "/tmp/hyperctl-batch-data/job2"
    assert job2.execution.working_dir == "/tmp"

    assert batch.backend_conf.type == 'local'
    assert batch.server_conf.host == 'local_machine'
    assert batch.server_conf.port == 18060
    assert batch.server_conf.exit_on_finish is False


