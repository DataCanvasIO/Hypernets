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


def test_batch_to_config():
    # 1. create a batch
    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")

    batch = Batch(generate_short_id(), batches_data_dir,
                  backend_conf=BackendConf(type='local'),
                  server_conf=ServerConf('localhost', 8063, exit_on_finish=True))

    data_dir = (Path(batches_data_dir)/"job1").absolute()
    execution_conf = ExecutionConf(command="pwd", data_dir=data_dir, working_dir=data_dir)

    job_params = {"learning_rate": 0.1}
    batch.add_job(name='job1', params=job_params, resource=None, execution=execution_conf)

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
    assert execution_conf_dict['data_dir'] == data_dir
    assert execution_conf_dict['working_dir'] == data_dir

    # 3.2 check backend
    backend_config = batch_config_dict['backend']
    assert backend_config['type'] == 'local'

    # 3.3 check server config
    server_config = batch_config_dict['server']
    assert server_config['host'] == 'localhost'
    assert server_config['port'] == 8063
    assert server_config['exit_on_finish'] is True
