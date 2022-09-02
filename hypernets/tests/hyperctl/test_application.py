import os
import tempfile
from pathlib import Path

from hypernets.hyperctl.appliation import BatchApplication
from hypernets.hyperctl.batch import _ShellJob
from hypernets.hyperctl.consts import JOB_DATA_DIR_PREFIX
from hypernets.hyperctl.executor import RemoteSSHExecutorManager, SSHRemoteMachine, LocalExecutorManager
from hypernets.hyperctl.utils import load_json

HERE = os.path.dirname(__file__)


def test_load_remote_batch():

    remote_batch_file = os.path.join(HERE, "remote_batch.json")
    batch_config_dict = load_json(remote_batch_file)

    batch_app = BatchApplication.load(batch_config_dict, "/tmp/test_remote_batch")
    print(batch_app)
    batch = batch_app.batch
    assert batch_app.batch.name == "remote-batch-example"
    assert batch_app.batch.data_dir_path.as_posix() == f"/tmp/test_remote_batch"

    assert len(batch.jobs) == 2
    job1: _ShellJob = batch.jobs[0]
    assert job1.name == "job1"
    assert job1.params == { "learning_rate": 0.1}

    # assert job1.data_dir_path.as_posix().startswith(f"/tmp/{JOB_DATA_DIR_PREFIX}{batch_app.batch.name}")
    assert job1.working_dir == job1.data_dir_path.as_posix()

    executor_manager = batch_app.job_scheduler.executor_manager
    assert isinstance(executor_manager, RemoteSSHExecutorManager)

    assert len(executor_manager.machines) == 2
    machine: SSHRemoteMachine = executor_manager.machines[0]
    assert machine.connection == {"hostname": "host1", "username": "hyperctl", "password": "hyperctl"}
    assert machine.environments == {"JAVA_HOME": "/usr/local/jdk"}

    job_scheduler = batch_app.job_scheduler
    assert job_scheduler.interval == 5000
    assert not job_scheduler.exit_on_finish

    assert batch_app.server_host == "localhost"
    assert batch_app.server_port == 8061


def test_load_local_batch_config():
    # 1. load batch from config
    local_batch = {
        "name": "local_batch_test",
        "job_command": "pwd",
        "jobs": [
            {
                "name": "job1",
                "params": {
                    "learning_rate": 0.1
                },

            }, {
                "name": "job2",
                "params": {
                    "learning_rate": 0.2
                }
            }
        ],
        "backend": {
            "type": "local",
            "conf": {}
        },
        "scheduler": {
            "exit_on_finish": False,
            "interval": 1
        },
        "server": {
            "host": "local_machine",
            "port": 18060
        }
    }
    batch_working_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")

    batch_app = BatchApplication.load(local_batch, batch_working_dir)

    # 2. assert batch
    assert batch_app.batch.name == "local_batch_test"
    jobs = batch_app.batch.jobs

    assert len(jobs) == 2
    job1: _ShellJob = jobs[0]
    assert isinstance(job1, _ShellJob)
    assert job1.name == "job1"
    assert job1.params['learning_rate'] == 0.1

    # assert job1.data_dir_path.as_posix().startswith(f"/tmp/{JOB_DATA_DIR_PREFIX}{batch_app.batch.name}")
    # assert job1.working_dir.startswith(f"/tmp/{JOB_DATA_DIR_PREFIX}{batch_app.batch.name}")

    job2: _ShellJob = jobs[1]
    assert isinstance(job2, _ShellJob)
    assert job2.name == "job2"
    assert job2.params['learning_rate'] == 0.2

    # check backend
    assert isinstance(batch_app.job_scheduler.executor_manager, LocalExecutorManager)

    # check server
    assert batch_app.server_host == 'local_machine'
    assert batch_app.server_port == 18060

    # check scheduler
    assert batch_app.job_scheduler.exit_on_finish is False
    assert batch_app.job_scheduler.interval == 1
