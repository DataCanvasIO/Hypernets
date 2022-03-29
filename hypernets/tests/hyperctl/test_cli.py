import os
import sys
import tempfile
from pathlib import Path
import _thread
import pytest
import time

from hypernets.hyperctl import api, get_context
from hypernets.hyperctl import cli, utils
from hypernets.hyperctl.batch import ShellJob, Batch
from hypernets.hyperctl.executor import LocalExecutorManager, RemoteSSHExecutorManager
from hypernets.tests.utils import ssh_utils_test
from hypernets.utils import is_os_windows


def test_run_generate_job_specs():
    batch_config_path = "hypernets/tests/hyperctl/remote_batch.yml"
    fd, fp = tempfile.mkstemp(prefix="jobs_spec_", suffix=".json")
    os.close(fd)
    os.remove(fp)

    cli.run_generate_job_specs(batch_config_path, fp)
    fp_ = Path(fp)

    assert fp_.exists()
    jobs_spec = utils.load_json(fp)
    assert len(jobs_spec['jobs']) == 4
    assert 'daemon' in jobs_spec
    assert 'name' in jobs_spec
    assert len(jobs_spec['backend']['conf']['machines']) == 2
    os.remove(fp_)
