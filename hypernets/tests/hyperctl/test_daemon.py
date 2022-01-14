import os
import sys
import tempfile
from pathlib import Path
import _thread
import pytest
import time

from hypernets.hyperctl import runtime, get_context
from hypernets.hyperctl import daemon
from hypernets.hyperctl.batch import ShellJob, Batch
from hypernets.hyperctl.executor import LocalExecutorManager, RemoteSSHExecutorManager
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
    spec_file_path = batch.spec_file_path()
    assert batch.spec_file_path().exists()
    assert daemon.load_json(spec_file_path)

    # pid file exists
    pid_file_path = batch.pid_file_path()
    assert pid_file_path.exists()

    # job succeed
    for job in batch.jobs:
        job: ShellJob = job
        assert Path(job.status_file_path(status)).as_posix()


def test_run_generate_job_specs():
    batch_config_path = "hypernets/tests/hyperctl/remote-example.yml"
    fd, fp = tempfile.mkstemp(prefix="jobs_spec_", suffix=".json")
    os.close(fd)
    os.remove(fp)

    daemon.run_generate_job_specs(batch_config_path, fp)
    fp_ = Path(fp)

    assert fp_.exists()
    jobs_spec = daemon.load_json(fp)
    assert len(jobs_spec['jobs']) == 4
    assert 'daemon' in jobs_spec
    assert 'name' in jobs_spec
    assert len(jobs_spec['backend']['conf']['machines']) == 2
    os.remove(fp_)


@ssh_utils_test.need_ssh
def test_run_remote():
    job1_name = "eVqNV5Uo1"
    job2_name = "eVqNV5Uo2"
    batch_name = "eVqNV5Ut"
    jobs_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-jobs")
    config_dict = {
        "jobs": [
            {
                "name": job1_name,
                "params": {
                    "learning_rate": 0.1
                },
                "resource": {
                },
                "execution": {
                    "command": "sleep 3",
                    "working_dir": "/tmp",
                    "data_dir": jobs_data_dir
                }
            }, {
                "name": job2_name,
                "params": {
                    "learning_rate": 0.1
                },
                "resource": {
                },
                "execution": {
                    "command": "sleep 3",
                    "working_dir": "/tmp",
                    "data_dir": jobs_data_dir
                }
            }
        ],
        "backend": {
            "type": "remote",
            "conf": {
                "machines": ssh_utils_test.get_ssh_test_config(use_password=True, use_rsa_file=False)
            }
        },
        "name": batch_name,
        "daemon": {
            "port": 8060,
            "exit_on_finish": True
        },
        "version": 2.5
    }

    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")

    daemon.run_batch(config_dict, batches_data_dir)
    executor_manager = get_context().executor_manager
    batch = get_context().batch
    assert isinstance(executor_manager, RemoteSSHExecutorManager)
    assert len(executor_manager.machines) == 2

    assert_batch_finished(batch, batch_name, [job1_name, job2_name], ShellJob.STATUS_SUCCEED)


@skip_if_windows
def test_run_local():
    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    job_name0 = "eVqNV5Uo0"
    job_name1 = "eVqNV5Uo1"
    batch_name = "eVqNV5Ut"
    local_example_script = Path("hypernets/tests/hyperctl/local-example-script.py").absolute()
    print(local_example_script)
    config_dict = {
        "jobs": [
            {
                "name": job_name0,
                "params": {
                    "learning_rate": 0.1
                },
                "resource": {
                    "cpu": 2
                },
                "execution": {
                    "command": f"{sys.executable} {local_example_script}",
                    "working_dir": "/tmp"
                }
            }, {
                "name": job_name1,
                "params": {
                    "learning_rate": 0.1
                },
                "resource": {
                    "cpu": 2
                },
                "execution": {
                    "command": f"{sys.executable} {local_example_script}",
                    "working_dir": "/tmp",
                }
            }
        ],
        "backend": {
            "type": "local",
            "conf": {}
        },
        "name": batch_name,
        "daemon": {
            "port": 8061,
            "exit_on_finish": True
        },
        "version": 2.5
    }

    print("Config:")
    print(config_dict)

    daemon.run_batch(config_dict, batches_data_dir)

    executor_manager = get_context().executor_manager
    assert isinstance(executor_manager, LocalExecutorManager)

    assert_batch_finished(get_context().batch, batch_name, [job_name0, job_name1], ShellJob.STATUS_SUCCEED)


@skip_if_windows
def test_kill_local_job():
    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")
    job_name = "eVqNV5Uo0"
    batch_name = "eVqNV5Ut"
    daemon_port = 8062

    config_dict = {
        "jobs": [
            {
                "name": job_name,
                "params": {
                    "learning_rate": 0.1
                },
                "resource": {
                },
                "execution": {
                    "command": "sleep 8",
                    "working_dir": "/tmp"
                }
            }
        ],
        "backend": {
            "type": "local",
            "conf": {}
        },
        "name": batch_name,
        "daemon": {
            "port": daemon_port,
            "exit_on_finish": True
        },
        "version": 2.5
    }

    def send_kill_request():
        time.sleep(6)
        runtime.kill_job(f'http://localhost:{daemon_port}', job_name)

    _thread.start_new_thread(send_kill_request, ())

    print("Config:")
    print(config_dict)

    daemon.run_batch(config_dict, batches_data_dir)
    batch = get_context().batch
    assert_batch_finished(get_context().batch, batch_name, [job_name], ShellJob.STATUS_FAILED)
    assert_local_job_finished(batch.jobs)


@skip_if_windows
def test_run_local_minimum_conf():
    batches_data_dir = tempfile.mkdtemp(prefix="hyperctl-test-batches")

    config_dict = {
        "jobs": [
            {
                "params": {
                    "learning_rate": 0.1
                },
                "execution": {
                    "command": "pwd"
                }
            }
        ],
        "daemon": {'exit_on_finish': True, 'port': 8063}
    }
    print("Config:")
    print(config_dict)

    daemon.run_batch(config_dict, batches_data_dir)

    executor_manager = get_context().executor_manager
    batch = get_context().batch
    batch_name = batch.name
    jobs_name = [j.name for j in batch.jobs]

    assert isinstance(executor_manager, LocalExecutorManager)

    assert_batch_finished(batch, batch_name, jobs_name, ShellJob.STATUS_SUCCEED)

    assert_local_job_succeed(batch.jobs)
