from pathlib import Path
import os
import tempfile

import pytest
from hypernets.utils import ssh_utils, common as common_util


def load_ssh_psw_config():
    return {
        'hostname': os.getenv('HYPERCTL_TEST_SSH_HOSTNAME_0'),
        'username': os.getenv('HYPERCTL_TEST_SSH_USERNAME_0'),
        'password': os.getenv('HYPERCTL_TEST_SSH_PASSWORD_0'),
    }


def load_ssh_rsa_config():
    return {
        'hostname': os.getenv('HYPERCTL_TEST_SSH_HOSTNAME_1'),
        'username': os.getenv('HYPERCTL_TEST_SSH_USERNAME_1'),
        'ssh_rsa_file': os.getenv('HYPERCTL_TEST_SSH_RSA_FILE_1'),
    }


def _is_all_values_not_none(dict_data):
    for v in dict_data.values():
        if v is None:
            return False
    return True


need_psw_auth_ssh = pytest.mark.skipif(not _is_all_values_not_none(load_ssh_psw_config()),
                                       reason='password authentication ssh account not ready,'
                                              ' please set ssh account in environment')
need_rsa_auth_ssh = pytest.mark.skipif(not _is_all_values_not_none(load_ssh_rsa_config()),
                                       reason='rsa file authentication ssh account not ready,'
                                              ' please set ssh account in environment ')


@need_psw_auth_ssh
def test_connect_by_password():
    client = ssh_utils.create_ssh_client(**load_ssh_psw_config())
    client.exec_command("pwd")
    assert client


@need_rsa_auth_ssh
def test_connect_by_rsa():
    client = ssh_utils.create_ssh_client(**load_ssh_rsa_config())
    client.exec_command("pwd")
    assert client


@need_psw_auth_ssh
def test_upload():
    ssh_config = load_ssh_psw_config()
    # generate temp file
    fd, fp = tempfile.mkstemp()
    os.close(fd)

    # upload
    with ssh_utils.sftp_client(**ssh_config) as client:
        p1 = common_util.generate_short_id()
        p2 = common_util.generate_short_id()
        r_path = (Path("/tmp") / p1 / p2 / Path(fp).name).as_posix()

        # check file in remote
        ssh_utils.copy_from_local_to_remote(client, fp, r_path)
        assert ssh_utils.exists(client, r_path)


@need_psw_auth_ssh
def test_makedirs():
    ssh_config = load_ssh_psw_config()
    with ssh_utils.sftp_client(**ssh_config) as client:
        p1 = common_util.generate_short_id()
        p2 = common_util.generate_short_id()
        r_path = (Path("/tmp") / p1 / p2).as_posix()
        print(f"made {r_path}")
        assert not ssh_utils.exists(client, r_path)
        ssh_utils.makedirs(client, r_path)
        assert ssh_utils.exists(client, r_path)


@need_psw_auth_ssh
def test_exec():
    ssh_config = load_ssh_psw_config()
    with ssh_utils.ssh_client(**ssh_config) as client:
        # transport = client.get_transport()
        # transport.set_keepalive(1)
        stdin, stdout, stderr = client.exec_command('cat /etc/hosts', get_pty=True)
        assert stdout.channel.recv_exit_status() == 0
        assert stdout.channel.exit_status_ready() is True
        hosts = stdout.read()
        assert len(hosts) > 0
