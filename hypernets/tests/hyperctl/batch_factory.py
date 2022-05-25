import os
import sys
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


def _create_local_batch(batch_name, job_assets=None):

    job1_name = "job1"
    job2_name = "job2"

    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    batch = Batch(batch_name, batches_data_dir)

    job1_data_dir = (batch.data_dir_path() / job1_name).absolute().as_posix()

    batch.add_job(name=job1_name,
                  params={"learning_rate": 0.1},
                  command=f"ls -l",
                  output_dir=job1_data_dir,
                  working_dir=job1_data_dir,
                  assets=job_assets)

    job2_data_dir = (Path(batches_data_dir) / job2_name).absolute().as_posix()
    batch.add_job(name=job2_name,
                  params={"learning_rate": 0.2},
                  command=f"ls -l",
                  output_dir=job2_data_dir,
                  working_dir=job2_data_dir)

    return batch


def create_local_batch(job_assets=None):
    return _create_local_batch("local-batch", job_assets)


def create_remote_batch(job_asserts=None):
    return _create_local_batch(batch_name="remote-batch", job_assets=job_asserts)


def create_assert_env_batch():
    job1_name = "job1"

    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    batch = Batch("assert_env_batch", batches_data_dir)

    job1_data_dir = (batch.data_dir_path() / job1_name).absolute().as_posix()
    py_code = f"import os; ch=os.environ['hyn_test_conda_home']; print(ch); assert ch == '/home/hyperctl/miniconda3' "
    batch.add_job(name=job1_name,
                  params={"learning_rate": 0.1},
                  command=f"{sys.executable} -c \"{py_code}\"",
                  output_dir=job1_data_dir,
                  working_dir=job1_data_dir)

    return batch


def create_assets_batch(data_dir):
    # create a batch with assets
    job1_name = "job1"
    batch_name = "test_run_batch"
    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    batch = Batch(batch_name, batches_data_dir)

    job1_data_dir_path = (batch.data_dir_path() / job1_name).absolute()
    job1_data_dir = job1_data_dir_path.as_posix()
    job_asserts = [data_dir.as_posix()]

    batch.add_job(name=job1_name,
                  params={"learning_rate": 0.1},
                  command=f"cat resources/{data_dir.name}/sub_dir/b.txt",  # read files in remote
                  output_dir=job1_data_dir,
                  working_dir=job1_data_dir,
                  assets=job_asserts)
    return batch
