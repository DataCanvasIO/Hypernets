import tempfile
from pathlib import Path

from hypernets.hyperctl.appliation import BatchApplication
from hypernets.hyperctl.batch import _ShellJob
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

    assert batch_config_dict['job_command'] == 'pwd'

    # 3. asset config content
    # 3.1. check jobs
    jobs_config = batch_config_dict['jobs']
    assert len(jobs_config) == 1
    job_config = jobs_config[0]

    assert job_config['name'] == 'job1'
    assert job_config['params']["learning_rate"] == 0.1

    assert job_config['working_dir']

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


def test_get_job_by_name():
    batch = create_local_batch()
    req_job_name = "job2"
    job = batch.get_job_by_name(req_job_name)
    assert job.name == req_job_name
    assert batch.get_persisted_job_status(req_job_name) == _ShellJob.STATUS_INIT
    assert job.params['learning_rate'] == 0.2
