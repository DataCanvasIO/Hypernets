import os
import tempfile
from pathlib import Path

from hypernets.hyperctl.batch import Batch
from hypernets.tests.utils import ssh_utils_test

SRC_DIR = os.path.dirname(__file__)


def create_minimum_batch(command="pwd", batches_data_dir=None):
    if batches_data_dir is None:
        batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")

    batch = Batch("minimum-batch", batches_data_dir)

    data_dir = (Path(batches_data_dir)/ batch.name / "job1").absolute().as_posix()

    job_params = {"learning_rate": 0.1}
    batch.add_job(name='job1', params=job_params, resource=None, command=command, output_dir=data_dir, working_dir=data_dir)

    return batch


def _create_local_batch(batch_name):

    job1_name = "job1"
    job2_name = "job2"

    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    batch = Batch(batch_name, batches_data_dir)

    job1_data_dir = (batch.data_dir_path() / job1_name).absolute().as_posix()

    batch.add_job(name=job1_name,
                  params={"learning_rate": 0.1},
                  command=f"ls -l",
                  output_dir=job1_data_dir,
                  working_dir=job1_data_dir)

    job2_data_dir = (Path(batches_data_dir) / job2_name).absolute().as_posix()
    batch.add_job(name=job2_name,
                  params={"learning_rate": 0.2},
                  command=f"ls -l",
                  output_dir=job2_data_dir,
                  working_dir=job2_data_dir)

    return batch


def create_local_batch():
    return _create_local_batch("local-batch")


def create_remote_batch():
    return _create_local_batch(batch_name="remote-batch")
