from pathlib import Path
import os
import tempfile

import pytest
from hypernets.utils import ssh_utils, common as common_util


def load_ssh_test_config():
    """For test remote ssh, should set env like :
        HYPERCTL_TEST_SSH_HOSTNAME_1=<your_host>
        HYPERCTL_TEST_SSH_USERNAME_1=<your_user>
    """
    key_host = 'HYPERCTL_TEST_SSH_HOSTNAME'
    key_username = 'HYPERCTL_TEST_SSH_USERNAME'
    key_password = 'HYPERCTL_TEST_SSH_PASSWORD'
    key_rsa_file = 'HYPERCTL_TEST_SSH_RSA_FILE'

    hosts = []
    env_keys = []
    hosts_env_conf = []
    for i in [0, 1]:
        host_env_conf = {
            'hostname': f"{key_host}_{i}",
            'username': f"{key_username}_{i}",
            'password': f"{key_password}_{i}",
            'ssh_rsa_file': f"{key_rsa_file}_{i}",
        }
        hosts_env_conf.append(host_env_conf)
        env_keys.extend(host_env_conf.values())

    for host_env_conf in hosts_env_conf:
        host_conf = {k: os.environ.get(v) for k, v in host_env_conf.items()}
        hosts.append(host_conf)

    print("Read env keys:")
    print("\n".join(env_keys))
    return hosts


ssh_hosts_test_config = load_ssh_test_config()


def get_ssh_test_config(use_password=True, use_rsa_file=False):
    global ssh_hosts_test_config
    ssh_hosts_test_config_ = []
    for ssh_test_config in ssh_hosts_test_config:
        ssh_test_config_ = ssh_test_config.copy()
        if use_password:
            del ssh_test_config_['ssh_rsa_file']

        if use_rsa_file:
            del ssh_test_config_['password']
        ssh_hosts_test_config_.append(ssh_test_config_)

    return ssh_hosts_test_config_


def ssh_ready():
    global ssh_hosts_test_config
    for ssh_config in ssh_hosts_test_config:
        for v in ssh_config.values():
            if v is None:
                return False
    return True


need_ssh = pytest.mark.skipif(not ssh_ready(), reason='ssh accounts not ready ')


@need_ssh
def test_connect_by_password_or_rsafile():
    ssh_pwd_configs = get_ssh_test_config(use_password=True, use_rsa_file=False)
    # ssh_rsa_configs = get_ssh_test_config(use_password=False, use_rsa_file=True)  #

    ssh_configs = ssh_pwd_configs
    # ssh_configs.extend(ssh_rsa_configs)

    for ssh_config in ssh_configs:
        client = ssh_utils.create_ssh_client(**ssh_config)
        assert client


@need_ssh
def test_upload():
    ssh_config = get_ssh_test_config(use_password=True, use_rsa_file=False)[0]
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


@need_ssh
def test_makedirs():
    ssh_config = get_ssh_test_config(use_password=True, use_rsa_file=False)[0]
    with ssh_utils.sftp_client(**ssh_config) as client:
        p1 = common_util.generate_short_id()
        p2 = common_util.generate_short_id()
        r_path = (Path("/tmp") / p1 / p2).as_posix()
        print(f"made {r_path}")
        assert not ssh_utils.exists(client, r_path)
        ssh_utils.makedirs(client, r_path)
        assert ssh_utils.exists(client, r_path)


@need_ssh
def test_exec():
    ssh_config = get_ssh_test_config(use_password=True, use_rsa_file=False)[0]
    with ssh_utils.ssh_client(**ssh_config) as client:
        # transport = client.get_transport()
        # transport.set_keepalive(1)
        stdin, stdout, stderr = client.exec_command('cat /etc/hosts', get_pty=True)
        assert stdout.channel.recv_exit_status() == 0
        assert stdout.channel.exit_status_ready() is True
        hosts = stdout.read()
        assert len(hosts) > 0
