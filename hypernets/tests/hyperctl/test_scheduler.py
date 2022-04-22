import tempfile
import time
import threading

from pathlib import Path
import pytest
import asyncio
from hypernets.hyperctl import api
from hypernets.hyperctl import scheduler, utils
from hypernets.hyperctl.appliation import BatchApplication
from hypernets.hyperctl.batch import ShellJob, Batch
from hypernets.hyperctl.callbacks import ConsoleCallback
from hypernets.hyperctl.executor import LocalExecutorManager, RemoteSSHExecutorManager
from hypernets.tests.hyperctl.batch_factory import create_minimum_batch, create_local_batch, create_remote_batch
from hypernets.tests.utils import ssh_utils_test
from hypernets.tests.utils.ssh_utils_test import BaseUpload
from hypernets.utils import is_os_windows

skip_if_windows = pytest.mark.skipif(is_os_windows, reason='not test on windows now')  # not generate run.bat now


class BatchRunner(threading.Thread):

    def __init__(self, batch_app):
        super(BatchRunner, self).__init__()
        self.batch_app = batch_app

    def run(self) -> None:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        self.batch_app.start()

    def stop(self):
        self.batch_app.stop()


def assert_local_job_finished(jobs):
    rets = []
    for job in jobs:
        job_output_dir = job.output_dir
        # stdout is not None but stderr is None
        stdout = Path(job_output_dir) / "stdout"
        stderr = Path(job_output_dir) / "stderr"
        run_sh = Path(job_output_dir) / "run.sh"
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
        Path(batch.job_status_file_path(job_name=job.name, status=status)).exists()


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


@ssh_utils_test.need_psw_auth_ssh
class TestRunRemoteWithAssets(BaseUpload):

    def test_run_batch(self):
        # create a batch with assets
        job1_name = "job1"
        batch_name = "test_run_batch"
        batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
        batch = Batch(batch_name, batches_data_dir)

        job1_data_dir = (batch.data_dir_path() / job1_name).absolute().as_posix()
        job_asserts = [self.data_dir.as_posix()]

        batch.add_job(name=job1_name,
                      params={"learning_rate": 0.1},
                      command=f"cat resources/{self.data_dir.name}/sub_dir/b.txt",  # read files in remote
                      output_dir=job1_data_dir,
                      working_dir=job1_data_dir,
                      assets=job_asserts)

        backend_conf = {
            "machines": [ssh_utils_test.load_ssh_psw_config()]
        }
        app = BatchApplication(batch, server_port=8089,
                               backend_type='remote',
                               backend_conf=backend_conf,
                               scheduler_exit_on_finish=True,
                               scheduler_interval=1)
        app.start()
        job_scheduler = app.job_scheduler
        assert isinstance(job_scheduler.executor_manager, RemoteSSHExecutorManager)
        assert len(job_scheduler.executor_manager.machines) == 1

        assert_batch_finished(batch, batch.name, [batch.jobs[0].name], ShellJob.STATUS_SUCCEED)


@skip_if_windows
def test_run_minimum_local_batch():
    batch = create_minimum_batch()
    scheduler_callbacks = []
    app = BatchApplication(batch, server_port=8061,
                           scheduler_exit_on_finish=True,
                           scheduler_interval=1,
                           scheduler_callbacks=[ConsoleCallback()])
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

    app = BatchApplication(batch, server_port=server_port,
                           scheduler_exit_on_finish=True,
                           scheduler_interval=1)
    runner = BatchRunner(app)
    runner.start()
    time.sleep(2)
    api.kill_job(f'http://localhost:{server_port}', job_name)
    time.sleep(1)
    assert_batch_finished(batch, batch.name, [job_name], ShellJob.STATUS_FAILED)


@skip_if_windows
def test_stop_scheduler():
    batch = create_minimum_batch()
    app = BatchApplication(batch, server_port=8086,
                           scheduler_exit_on_finish=False,
                           scheduler_interval=1)
    runner = BatchRunner(app)
    runner.start()
    time.sleep(2)  # wait for starting
    assert runner.is_alive()
    runner.stop()
    time.sleep(1)
    assert not runner.is_alive()


def create_batch_app(batches_data_dir):

    batch = create_minimum_batch(batches_data_dir=batches_data_dir)

    app = BatchApplication(batch, server_port=8086,
                           scheduler_exit_on_finish=True,
                           scheduler_interval=1)

    return app


def test_run_base_previous_batch():
    # run a batch
    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    app1 = create_batch_app(batches_data_dir)
    app1.start()
    app1._http_server.stop()  # release port

    scheduler1 = app1.job_scheduler
    assert scheduler1.n_allocated == len(app1.batch.jobs)
    assert scheduler1.n_skipped == 0

    # run the bach base on previous batch
    app2 = BatchApplication.load(app1.to_config(), batches_data_dir=batches_data_dir)

    # app2 = create_batch_app(batches_data_dir)
    app2.start()
    app2._http_server.stop()
    scheduler2 = app2.job_scheduler

    # all ran jobs should not run again
    assert scheduler2.n_allocated == 0
    assert scheduler2.n_skipped == len(app1.batch.jobs)


# if __name__ == '__main__':
#     test_run_remote()
#
