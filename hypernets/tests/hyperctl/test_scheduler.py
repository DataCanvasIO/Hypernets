import tempfile
import time
import threading

from pathlib import Path
import pytest
import asyncio
from hypernets.hyperctl import api
from hypernets.hyperctl import scheduler, utils
from hypernets.hyperctl.appliation import BatchApplication
from hypernets.hyperctl.batch import _ShellJob, Batch
from hypernets.hyperctl.callbacks import ConsoleCallback
from hypernets.hyperctl.executor import LocalExecutorManager, RemoteSSHExecutorManager
from hypernets.tests.hyperctl import batch_factory
from hypernets.tests.hyperctl.batch_factory import create_minimum_batch, create_local_batch, create_remote_batch
from hypernets.tests.utils import ssh_utils_test
from hypernets.tests.utils.ssh_utils_test import BaseUpload, load_ssh_psw_config
from hypernets.utils import is_os_windows, ssh_utils

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


class BaseBatchAppTest:

    app = None

    @classmethod
    def setup_class(cls):
        # clear ioloop
        asyncio.set_event_loop(asyncio.new_event_loop())

    @classmethod
    def teardown_class(cls):
        if cls.app is not None:
            batch = cls.app.batch
            assert_batch_finished(batch, _ShellJob.STATUS_SUCCEED)
            cls.app.stop()
        # release resources
        asyncio.get_event_loop().stop()
        asyncio.get_event_loop().close()


def assert_local_job_finished(jobs):
    rets = []
    for job in jobs:
        job_data_dir = job.data_dir_path
        # stdout is not None but stderr is None
        stdout = Path(job_data_dir) / "stdout"
        stderr = Path(job_data_dir) / "stderr"
        run_sh = Path(job_data_dir) / "run.sh"
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


def assert_batch_finished(batch: Batch, status, input_batch_name=None, input_jobs_name=None):

    if input_batch_name is not None:
        assert batch.name == input_batch_name

    if input_jobs_name is not None:
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
        job: _ShellJob = job
        assert Path(batch.job_status_file_path(job_name=job.name, status=status)).exists()


def create_remote_backend_conf():
    return {
        "type": "remote",
        "machines": [{"connection": load_ssh_psw_config()}]
    }


@skip_if_windows
@ssh_utils_test.need_psw_auth_ssh
class TestRemoteBatch(BaseBatchAppTest):

    @classmethod
    def setup_class(cls):
        super(TestRemoteBatch, cls).setup_class()
        batch = create_remote_batch()

        backend_conf = create_remote_backend_conf()
        app = BatchApplication(batch, server_port=8088,
                               backend_conf=backend_conf,
                               scheduler_exit_on_finish=True,
                               scheduler_interval=1000)
        cls.app = app

    def test_run_batch(self):
        app = self.app
        app.start()
        job_scheduler = app.job_scheduler
        assert isinstance(job_scheduler.executor_manager, RemoteSSHExecutorManager)
        assert len(job_scheduler.executor_manager.machines) == 1


@ssh_utils_test.need_psw_auth_ssh
class TestRunRemoteWithAssets(BaseUpload):

    @classmethod
    def setup_class(cls):
        super(TestRunRemoteWithAssets, cls).setup_class()
        # clear ioloop
        asyncio.set_event_loop(asyncio.new_event_loop())

        # create a batch with assets
        batch = batch_factory.create_assets_batch(cls.data_dir)
        backend_conf = create_remote_backend_conf()
        app = BatchApplication(batch, server_port=8089,
                               backend_type='remote',
                               backend_conf=backend_conf,
                               scheduler_exit_on_finish=True,
                               scheduler_interval=1000)
        cls.app = app

    def test_run_batch(self):
        app = self.app
        batch = app.batch
        app.start()
        job_scheduler = app.job_scheduler
        assert isinstance(job_scheduler.executor_manager, RemoteSSHExecutorManager)
        assert len(job_scheduler.executor_manager.machines) == 1

        assert_batch_finished(batch, _ShellJob.STATUS_SUCCEED)

        # check assets in remote
        job1_data_dir_path = batch.jobs[0].data_dir_path
        with ssh_utils.sftp_client(**self.ssh_config) as client:
            remote_assert_path = job1_data_dir_path / "resources" / self.data_dir.name

            ssh_utils.exists(client, (remote_assert_path / "empty_dir").as_posix())
            ssh_utils.exists(client, (remote_assert_path / "a.txt").as_posix())
            ssh_utils.exists(client, (remote_assert_path / "sub_dir" / "b.txt").as_posix())

    @classmethod
    def teardown_class(cls):
        if cls.app is not None:
            cls.app.stop()

        # release resources
        asyncio.get_event_loop().stop()
        asyncio.get_event_loop().close()


@skip_if_windows
class TestMinimumLocalBatch(BaseBatchAppTest):

    @classmethod
    def setup_class(cls):
        super(TestMinimumLocalBatch, cls).setup_class()
        batch = create_minimum_batch()
        app = BatchApplication(batch, server_port=8061,
                               scheduler_exit_on_finish=True,
                               scheduler_interval=1000,
                               scheduler_callbacks=[ConsoleCallback()])
        cls.app = app

    def test_run_batch(self):
        app = self.app
        app.start()
        batch = app.batch
        assert_batch_finished(batch, _ShellJob.STATUS_SUCCEED)
        assert_local_job_succeed(batch.jobs)


@skip_if_windows
class TestLocalBatch(BaseBatchAppTest):
    @classmethod
    def setup_class(cls):
        super(TestLocalBatch, cls).setup_class()
        batch = create_local_batch()
        app = BatchApplication(batch, server_port=8082,
                               scheduler_exit_on_finish=True,
                               scheduler_interval=1000)
        cls.app = app

    def test_run_batch(self):
        app = self.app
        app.start()
        batch = app.batch
        assert isinstance(app.job_scheduler.executor_manager, LocalExecutorManager)
        assert_batch_finished(batch, _ShellJob.STATUS_SUCCEED)
        assert_local_job_succeed(batch.jobs)


@skip_if_windows
class TestKillLocalJob(BaseBatchAppTest):

    @classmethod
    def setup_class(cls):
        super(TestKillLocalJob, cls).setup_class()
        batch = create_minimum_batch(command='sleep 6; echo "finished"')

        server_port = 8063
        cls.server_port = server_port

        app = BatchApplication(batch, server_port=server_port,
                               scheduler_exit_on_finish=True,
                               scheduler_interval=1000)
        runner = BatchRunner(app)
        cls.app = app
        cls.runner = runner
        runner.start()

    def test_run_batch(self):
        job_name = self.app.batch.jobs[0].name
        time.sleep(2)
        api.kill_job(f'http://localhost:{self.server_port}', job_name)
        time.sleep(1)

    @classmethod
    def teardown_class(cls):
        if cls.app is not None:
            batch = cls.app.batch
            assert_batch_finished(batch, _ShellJob.STATUS_FAILED)
            cls.app.stop()

        if cls.runner is not None:
            cls.runner.stop()
        # release resources
        asyncio.get_event_loop().stop()
        asyncio.get_event_loop().close()



@skip_if_windows
class TestStopScheduler(BaseBatchAppTest):
    @classmethod
    def setup_class(cls):
        super(TestStopScheduler, cls).setup_class()
        batch = create_minimum_batch()
        app = BatchApplication(batch, server_port=8086,
                               scheduler_exit_on_finish=False,
                               scheduler_interval=1000)
        runner = BatchRunner(app)
        runner.start()
        cls.runner = runner
        time.sleep(2)  # wait for starting

    def run_stop_scheduler(self):
        runner = self.runner
        assert runner.is_alive()
        runner.stop()
        time.sleep(1)
        assert not runner.is_alive()

    @classmethod
    def teardown_class(cls):
        if cls.runner is not None:
            cls.runner.stop()
        super(TestStopScheduler, cls).teardown_class()


def create_batch_app(batches_data_dir):

    batch = create_minimum_batch(batches_data_dir=batches_data_dir)

    app = BatchApplication(batch, server_port=8086,
                           scheduler_exit_on_finish=True,
                           scheduler_interval=1000)

    return app


class TestRunBasePreviousBatch(BaseBatchAppTest):

    @classmethod
    def setup_class(cls):
        super(TestRunBasePreviousBatch, cls).setup_class()

        # run a batch
        batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
        app1 = create_batch_app(batches_data_dir)

        app1.start()
        app1._http_server.stop()  # release port
        cls.app1 = app1
        scheduler1 = app1.job_scheduler
        assert scheduler1.n_allocated == len(app1.batch.jobs)
        assert scheduler1.n_skipped == 0

    def test_run_batch(self):
        app1 = self.app1
        # run the bach base on previous batch
        app2 = BatchApplication.load(app1.to_config(), batch_data_dir=app1.batch.data_dir_path)

        # app2 = create_batch_app(batches_data_dir)
        app2.start()
        app2._http_server.stop()
        scheduler2 = app2.job_scheduler

        # all ran jobs should not run again
        assert scheduler2.n_allocated == 0
        assert scheduler2.n_skipped == len(app1.batch.jobs)


@skip_if_windows
class TestLocalHostEnv(BaseBatchAppTest):

    @classmethod
    def setup_class(cls):
        super(TestLocalHostEnv, cls).setup_class()

        batch = batch_factory.create_assert_env_batch()
        app = BatchApplication(batch, server_port=8088,
                               scheduler_exit_on_finish=True,
                               backend_type='local',
                               backend_conf=dict(environments={"hyn_test_conda_home": "/home/hyperctl/miniconda3"}),
                               scheduler_interval=1000)
        cls.app = app

    def test_run_batch(self):
        app = self.app

        self.app.start()
        assert isinstance(app.job_scheduler.executor_manager, LocalExecutorManager)
        assert_batch_finished(app.batch, _ShellJob.STATUS_SUCCEED)


@skip_if_windows
@ssh_utils_test.need_psw_auth_ssh
class TestRemoteHostEnv(BaseBatchAppTest):

    @classmethod
    def setup_class(cls):
        super(TestRemoteHostEnv, cls).setup_class()
        backend_conf = {
            "type": 'remote',
            "machines": [{
                'connection':  load_ssh_psw_config(),
                'environments': {"hyn_test_conda_home": "/home/hyperctl/miniconda3"}
            }]
        }
        batch = batch_factory.create_assert_env_batch()
        app = BatchApplication(batch, server_port=8089,
                               scheduler_exit_on_finish=True,

                               backend_conf=backend_conf,
                               scheduler_interval=1000)
        cls.app = app

    def test_run_batch(self):
        app = self.app

        self.app.start()
        assert isinstance(app.job_scheduler.executor_manager, RemoteSSHExecutorManager)
        assert_batch_finished(app.batch, _ShellJob.STATUS_SUCCEED)


@skip_if_windows
class TestJobCache(BaseBatchAppTest):

    @staticmethod
    def create_app(batches_data_dir, port):
        batch = batch_factory.create_local_batch(batches_data_dir=batches_data_dir)
        app = BatchApplication(batch, server_port=port,
                               scheduler_exit_on_finish=True,
                               backend_conf={'type': 'local'},
                               scheduler_interval=1000)
        return app

    @classmethod
    def setup_class(cls):
        super(TestJobCache, cls).setup_class()
        batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
        cls.app = cls.create_app(batches_data_dir, 8090)
        cls.app1 = cls.create_app(batches_data_dir, 8091)

    def test_run_batch(self):

        self.app1.start()

        self.app.start()

        assert isinstance(self.app.job_scheduler.executor_manager, LocalExecutorManager)
        assert_batch_finished(self.app.batch, _ShellJob.STATUS_SUCCEED)

        assert isinstance(self.app1.job_scheduler.executor_manager, LocalExecutorManager)
        assert_batch_finished(self.app1.batch, _ShellJob.STATUS_SUCCEED)

        assert self.app.job_scheduler.n_skipped == 2


# @skip_if_windows
# class TestCallbackBatch(BaseBatchAppTest):
#
#     @classmethod
#     def setup_class(cls):
#         super(TestMinimumLocalBatch, cls).setup_class()
#         batch = create_minimum_batch()
#         app = BatchApplication(batch, server_port=8061,
#                                scheduler_exit_on_finish=True,
#                                scheduler_interval=1000,
#                                scheduler_callbacks=[ConsoleCallback()])
#         cls.app = app
#
#     def test_run_batch(self):
#         app = self.app
#         app.start()
#         batch = app.batch
#         assert_batch_finished(batch, _ShellJob.STATUS_SUCCEED)
#         assert_local_job_succeed(batch.jobs)
