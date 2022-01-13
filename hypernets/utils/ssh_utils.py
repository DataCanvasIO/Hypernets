import paramiko
import contextlib
import os
from pathlib import Path

from paramiko import SFTPClient, SSHClient


def create_ssh_client(hostname,  username, port=22, password=None, ssh_rsa_file=None, passphrase=None) -> SSHClient:
    client = paramiko.SSHClient()
    # auto-save server info to local know_hosts or can not connect to server that not in know_hosts
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs = {'hostname': hostname, 'username': username, 'port': port}
    if ssh_rsa_file is not None:
        kwargs['key_filename'] = ssh_rsa_file

    if password is not None:
        kwargs['password'] = password

    if passphrase is not None:
        kwargs['passphrase'] = passphrase
    client.connect(**kwargs)
    return client


@contextlib.contextmanager
def ssh_client(*args, **kwargs):
    client: SSHClient = create_ssh_client(*args, **kwargs)
    yield client
    if client is not None:
        client.close()


def create_sftp_client(*args, **kwargs):
    return create_ssh_client(*args, **kwargs).open_sftp()


@contextlib.contextmanager
def sftp_client(*args, **kwargs):
    _sftp_client: SFTPClient = create_sftp_client(*args, **kwargs)
    yield _sftp_client
    if _sftp_client is not None:
        _sftp_client.close()


def exists(sftp: SFTPClient, remote_path):
    try:
        sftp.lstat(remote_path)
        return True
    except FileNotFoundError as e:
        return False
    except Exception as e:
        raise e


def makedirs(sftp: SFTPClient, remote_dir):
    if exists(sftp, remote_dir):
        return

    # check parent dir
    pp = Path(remote_dir).parent.as_posix()

    if exists(sftp, pp):
        sftp.mkdir(remote_dir)
    else:
        makedirs(sftp, pp)
        sftp.mkdir(remote_dir)


def copy_from_local_to_remote(sftp: SFTPClient, local_path, remote_path):
    p = Path(remote_path).parent.as_posix()
    makedirs(sftp, p)
    sftp.put(local_path, remote_path)
