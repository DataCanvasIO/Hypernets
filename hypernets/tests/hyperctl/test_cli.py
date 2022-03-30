import os
import tempfile
from pathlib import Path

from hypernets.hyperctl import cli, utils


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
