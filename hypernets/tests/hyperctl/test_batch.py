import tempfile
from pathlib import Path

from hypernets.hyperctl.appliation import BatchApplication
from hypernets.hyperctl.batch import ShellJob
from hypernets.hyperctl.executor import LocalExecutorManager, RemoteSSHExecutorManager
from hypernets.tests.hyperctl.batch_factory import create_minimum_batch, create_local_batch


def test_batch_to_config():
    server_port = 8061
    scheduler_interval = 1
    # 1. create a batch
    batch = create_minimum_batch()
    app = BatchApplication(batch, server_port=server_port,
                           scheduler_exit_on_finish=True,
                           scheduler_interval=scheduler_interval)

    # 2. to_config
    batch_config_dict = app.to_config()

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

    # 3.2 TODO check backend
    # backend_config = batch_config_dict['backend']
    # assert backend_config['type'] == 'local'

    # 3.3 check server config
    server_config = batch_config_dict['server']
    assert server_config['host'] == 'localhost'
    assert server_config['port'] == server_port

    # 3.4 check scheduler
    scheduler_config = batch_config_dict['scheduler']
    assert scheduler_config['exit_on_finish'] is True
    assert scheduler_config['interval'] == 1

    # 3.4. check version
    assert batch_config_dict['version']


def test_load_local_batch_config():
    # 1. load batch from config
    job2_data_dir = "/tmp/hyperctl-batch-data/job2"
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
                    "working_dir": job2_data_dir,
                    "data_dir": job2_data_dir,
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
    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")

    batch_app = BatchApplication.load(local_batch, batches_data_dir)

    # 2. assert batch
    assert batch_app.batch.name == "local_batch_test"
    jobs = batch_app.batch.jobs

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
    assert job2.execution.data_dir == job2_data_dir
    assert job2.execution.working_dir == job2_data_dir

    # check backend
    assert isinstance(batch_app.job_scheduler.executor_manager, LocalExecutorManager)

    # check server
    assert batch_app.server_host == 'local_machine'
    assert batch_app.server_port == 18060

    # check scheduler
    assert batch_app.job_scheduler.exit_on_finish is False
    assert batch_app.job_scheduler.interval == 1


def test_load_remote_batch_config():
    # 1. load batch from config
    job2_data_dir = "/tmp/hyperctl-batch-data/job2"
    local_batch = {
        "name": "remote_batch_test",
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
                    "working_dir": job2_data_dir,
                    "data_dir": job2_data_dir,
                }
            }
        ],
        "backend": {
            "type": "remote",
            "conf": {
                "machines": [
                    {
                        "hostname": "host1",
                        "username": "hyperctl",
                        "password": "hyperctl"
                    },
                    {
                        "hostname": "host2",
                        "username": "hyperctl",
                        "password": "hyperctl"
                    }
                ]
            }
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
    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    batch_app = BatchApplication.load(local_batch, batches_data_dir)

    # check backend
    executor_manager: RemoteSSHExecutorManager = batch_app.job_scheduler.executor_manager
    assert isinstance(executor_manager, RemoteSSHExecutorManager)

    assert len(executor_manager.machines) == 2
    machine1 = executor_manager.machines[0]
    assert machine1.hostname == "host1"

    machine2 = executor_manager.machines[1]
    assert machine2.hostname == "host2"


def test_get_job_by_name():
    batch = create_local_batch()
    req_job_name = "job2"
    job = batch.get_job_by_name(req_job_name)
    assert job.name == req_job_name
    assert job.status == ShellJob.STATUS_INIT
    assert job.params['learning_rate'] == 0.2
