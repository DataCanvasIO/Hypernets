import os
import tempfile
from pathlib import Path

from hypernets.hyperctl.batch import BackendConf, ServerConf, ExecutionConf
from hypernets.hyperctl.batch import Batch
from hypernets.tests.utils import ssh_utils_test

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


def _create_local_batch(batch_name, backend=None):
    if backend is None:
        backend = BackendConf(type='local')

    job1_name = "job1"
    job2_name = "job2"
    daemon_port = 8062

    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    batch = Batch(batch_name, batches_data_dir,
                  backend_conf=backend,
                  server_conf=ServerConf('localhost', daemon_port, exit_on_finish=True))

    job1_data_dir = (Path(batches_data_dir)/ job1_name).absolute().as_posix()



    batch.add_job(name=job1_name,
                  params={"learning_rate": 0.1},
                  execution=ExecutionConf(command=f"ls -l",
                                          data_dir=job1_data_dir,
                                          working_dir=job1_data_dir))

    job2_data_dir = (Path(batches_data_dir) / job2_name).absolute().as_posix()
    batch.add_job(name=job2_name,
                  params={"learning_rate": 0.2},
                  execution=ExecutionConf(command=f"ls -l",
                                          data_dir=job2_data_dir,
                                          working_dir=job2_data_dir))
    return batch


def create_local_batch():
    return _create_local_batch("local-batch")


def create_remote_batch():

    backend_conf = BackendConf(type='remote', conf={
        "machines": [ssh_utils_test.load_ssh_psw_config()]
    })

    return _create_local_batch(batch_name="remote-batch", backend=backend_conf)
