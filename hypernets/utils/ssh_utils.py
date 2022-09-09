import paramiko
import contextlib
import os
from pathlib import Path
from hypernets.utils import logging as hyn_logging

from paramiko import SFTPClient, SSHClient

logger = hyn_logging.get_logger(__name__)


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


@contextlib.contextmanager
def sftp_client(*args, **kwargs):

    ssh_client_inst = create_ssh_client(*args, **kwargs)
    _sftp_client: SFTPClient = ssh_client_inst.open_sftp()
    yield _sftp_client

    if _sftp_client is not None:
        _sftp_client.close()

    if ssh_client_inst is not None:  # only close sftp client lead to connection leak
        ssh_client_inst.close()


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


def upload_file(sftp: SFTPClient, local_path, remote_path):
    """Copy local file to remote. if remote dir is not exists should create it.
    :param sftp:
    :param local_path: a file
    :param remote_path: a file path, is not exists and included file name
    :return:
    """
    p = Path(remote_path).parent.as_posix()
    makedirs(sftp, p)
    sftp.put(local_path, remote_path)


def upload_file_obj(sftp: SFTPClient, fo, remote_path):
    """Upload file object to remote. if remote dir is not exists should create it.
    :param sftp:
    :param fo: file obj
    :param remote_path: a file path, is not exists and included file name
    :return:
    """
    p = Path(remote_path).parent.as_posix()
    makedirs(sftp, p)
    sftp.putfo(fo, remote_path, confirm=True)


def upload_dir(sftp: SFTPClient, local_dir, remote_dir):
    """ Recursive upload local dir to remote
    :param sftp:
    :param local_dir: a local dir
    :param remote_dir: a not exists path in remote server
    :return:
    """

    local_dir_path = Path(local_dir)
    assert local_dir_path.exists() and local_dir_path.is_dir(), f"input local_dir {local_dir} should be a exists dir"

    remote_destination_dir_path = Path(remote_dir).absolute() / local_dir_path.name
    remote_destination_dir = remote_destination_dir_path.as_posix()

    assert not exists(sftp, remote_destination_dir), f"remote file {remote_destination_dir} path already existed "

    for walk_root, ds, fs in os.walk(local_dir_path):
        relative_walk_root_path = Path(walk_root).relative_to(local_dir_path)
        remote_walk_root = remote_destination_dir_path / relative_walk_root_path  # is a dir
        # create remote dir even is empty dir
        makedirs(sftp, remote_walk_root.as_posix())
        logger.debug(f"create remote fold {remote_walk_root.absolute()}")

        # upload files
        for f in fs:
            remote_file = (remote_walk_root / f).as_posix()
            local_file = (Path(walk_root) / f).as_posix()
            upload_file(sftp, local_file, remote_file)
            logger.debug(f"uploaded local file {local_file} to remote {remote_file}")

