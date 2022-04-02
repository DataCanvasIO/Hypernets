import _thread
import time
from pathlib import Path

import pytest

from hypernets.hyperctl import api
from hypernets.hyperctl import scheduler, utils
from hypernets.hyperctl.appliation import BatchApplication
from hypernets.hyperctl.batch import ShellJob, Batch
from hypernets.hyperctl.executor import LocalExecutorManager, RemoteSSHExecutorManager
from hypernets.tests.hyperctl.batch_factory import create_minimum_batch, create_local_batch, create_remote_batch
from hypernets.tests.utils import ssh_utils_test
from hypernets.utils import is_os_windows

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
    spec_file_path = batch.config_file_path()
    assert batch.config_file_path().exists()
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

    backend_conf = {
        "machines": [ssh_utils_test.load_ssh_psw_config()]
    }
    app = BatchApplication(batch, server_port=8088,
                           backend_type='remote',
                           backend_conf=backend_conf,
                           scheduler_exit_on_finish=True,
                           scheduler_interval=1)
    app.start()
    job_scheduler = app.job_scheduler
    assert isinstance(job_scheduler.executor_manager, RemoteSSHExecutorManager)
    assert len(job_scheduler.executor_manager.machines) == 1

    assert_batch_finished(batch, batch.name, [batch.jobs[0].name, batch.jobs[1].name], ShellJob.STATUS_SUCCEED)


@skip_if_windows
def test_run_minimum_local_batch():
    batch = create_minimum_batch()
    app = BatchApplication(batch, server_port=8061,
                           scheduler_exit_on_finish=True,
                           scheduler_interval=1)
    app.start()
    assert_batch_finished(batch, batch.name, [batch.jobs[0].name], ShellJob.STATUS_SUCCEED)
    assert_local_job_succeed(batch.jobs)


@skip_if_windows
def test_run_local_batch():
    batch = create_local_batch()
    app = BatchApplication(batch, server_port=8082,
                           scheduler_exit_on_finish=True,
                           scheduler_interval=1)
    app.start()
    assert isinstance(app.job_scheduler.executor_manager, LocalExecutorManager)
    assert_batch_finished(batch, batch.name, [batch.jobs[0].name, batch.jobs[1].name], ShellJob.STATUS_SUCCEED)
    assert_local_job_succeed(batch.jobs)


@skip_if_windows
def test_kill_local_job():
    batch = create_minimum_batch(command='sleep 6; echo "finished"')
    job_name = batch.jobs[0].name
    server_port = 8063

    def send_kill_request():
        time.sleep(2)
        api.kill_job(f'http://localhost:{server_port}', job_name)

    _thread.start_new_thread(send_kill_request, ())

    app = BatchApplication(batch, server_port=server_port,
                           scheduler_exit_on_finish=True,
                           scheduler_interval=1)
    app.start()
    assert_batch_finished(batch, batch.name, [job_name], ShellJob.STATUS_FAILED)


# if __name__ == '__main__':
#     test_run_remote()
