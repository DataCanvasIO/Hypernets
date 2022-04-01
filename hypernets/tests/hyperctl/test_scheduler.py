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
from hypernets.tests.hyperctl.batch_factory import create_minimum_batch, create_local_batch, create_remote_batch
from hypernets.tests.utils import ssh_utils_test
from hypernets.utils import is_os_windows
from hypernets.utils.common import generate_short_id

skip_if_windows = pytest.mark.skipif(is_os_windows, reason='not test on windows now')  # not generate run.bat now


def assert_local_job_finished(jobs):
    rets = []
    for job in jobs:
        job_execution_data_dir = job.execution.data_dir
        # stdout is not None but stderr is None
        stdout = Path(job_execution_data_dir) / "stdout"
        stderr = Path(job_execution_data_dir) / "stderr"
        run_sh = Path(job_execution_data_dir) / "run.sh"
        assert stdout.exists()
        assert stderr.exists()
        assert run_sh.exists()
        rets.append((stdout, stderr, run_sh))

    return rets


def assert_local_job_succeed(jobs):
    rets = assert_local_job_finished(jobs)
    for stdout, stderr, run_sh in rets:
        with open(stdout, 'r') as f:
            assert len(f.read()) > 0
        with open(stderr, 'r') as f:
            assert len(f.read()) == 0


def assert_batch_finished(batch: Batch, input_batch_name, input_jobs_name,  status):

    assert batch.name == input_batch_name
    assert set([job.name for job in batch.jobs]) == set(input_jobs_name)

    # spec.json exists and is json file
    spec_file_path = batch.spec_file_path()
    assert batch.spec_file_path().exists()
    assert utils.load_json(spec_file_path)

    # pid file exists
    pid_file_path = batch.pid_file_path()
    assert pid_file_path.exists()

    # assert job status
    for job in batch.jobs:
        job: ShellJob = job
        assert Path(job.status_file_path(status)).exists()


@ssh_utils_test.need_psw_auth_ssh
def test_run_remote():
    batch = create_remote_batch()

    job_scheduler = scheduler.run_batch(batch)
    assert isinstance(job_scheduler.executor_manager, RemoteSSHExecutorManager)
    assert len(job_scheduler.executor_manager.machines) == 2

    assert_batch_finished(batch, batch.name, [batch.jobs[0].name, batch.jobs[1].name], ShellJob.STATUS_SUCCEED)


@skip_if_windows
def test_run_minimum_local_batch():
    batch = create_minimum_batch()
    scheduler.run_batch(batch)
    assert_batch_finished(batch, batch.name, [batch.jobs[0].name], ShellJob.STATUS_SUCCEED)
    assert_local_job_succeed(batch.jobs)


@skip_if_windows
def test_run_local():
    batch = create_local_batch()
    # executor_manager = get_context().executor_manager
    # assert isinstance(executor_manager, LocalExecutorManager)
    scheduler.run_batch(batch)
    assert_batch_finished(batch, batch.name, [batch.jobs[0].name, batch.jobs[1].name], ShellJob.STATUS_SUCCEED)
    assert_local_job_succeed(batch.jobs)


@skip_if_windows
def test_kill_local_job():
    batch = create_minimum_batch(command='sleep 8; echo "finished"')
    job_name = batch.jobs[0].name

    def send_kill_request():
        time.sleep(6)
        api.kill_job(f'http://localhost:{8063}/hyperctl', job_name)

    _thread.start_new_thread(send_kill_request, ())
    scheduler.run_batch(batch)
    assert_batch_finished(batch, batch.name, [job_name], ShellJob.STATUS_FAILED)
    time.sleep(1)
