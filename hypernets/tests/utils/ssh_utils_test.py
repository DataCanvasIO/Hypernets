from pathlib import Path
import os
import io
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
        ssh_utils.upload_file(client, fp, r_path)
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


class BaseUpload:

    @classmethod
    def setup_class(cls):
        """Create files:
        test-dir/
        |-- a.txt
        |-- empty_dir
        |__ sub_dir
            |__ b.txt

        :return:
        """
        # create a folder
        test_dir_path = Path(tempfile.mkdtemp(prefix="ssh-util-tests"))

        cls.data_dir = test_dir_path
        print("local test dir ")
        print(test_dir_path)

        # create a file
        cls.file_a = (test_dir_path / "a.txt").as_posix()
        with open(cls.file_a, 'w') as f:
            f.write("this is a")

        # create empty_dir
        cls.file_subdir = (test_dir_path / "empty_dir").as_posix()
        os.makedirs(cls.file_subdir)

        # create sub-dir
        cls.file_subdir = (test_dir_path / "sub_dir").as_posix()
        os.makedirs(cls.file_subdir)

        # create a file in sub-dir
        cls.file_in_sub_dir = (Path(cls.file_subdir) / "b.txt").as_posix()
        with open(cls.file_in_sub_dir, 'w') as f:
            f.write("this is b")

        cls.ssh_config = load_ssh_psw_config()

    def upload_dir(self):
        with ssh_utils.sftp_client(**self.ssh_config) as client:
            # check file in remote
            p1 = common_util.generate_short_id()
            p2 = common_util.generate_short_id()
            remote_dir_path = (Path("/tmp") / p1 / p2)
            remote_dir = remote_dir_path.as_posix()

            ssh_utils.upload_dir(client, self.data_dir, remote_dir)
            return remote_dir



@need_psw_auth_ssh
class TestUpload(BaseUpload):

    def test_upload_file(self):
        # check file in remote
        p1 = common_util.generate_short_id()
        p2 = common_util.generate_short_id()
        remote_file = (Path("/tmp") / p1 / p2 / Path(self.file_a).name).as_posix()
        self.run_upload_file(remote_file)

    def test_upload_fo(self):
        # check file in remote
        p1 = common_util.generate_short_id()
        p2 = common_util.generate_short_id()
        remote_file = (Path("/tmp") / p1 / p2 / "fo.bin").as_posix()

        bytes_data = "abc123".encode('utf-8')
        fo = io.BytesIO(bytes_data)
        with ssh_utils.sftp_client(**self.ssh_config) as client:
            print(remote_file)
            ssh_utils.upload_file_obj(client, fo, remote_file)
            # check file in remote
            assert ssh_utils.exists(client, remote_file)

    def test_upload_file_to_exists_folder(self):
        remote_file = (Path("/tmp") / Path(self.file_a).name).as_posix()
        self.run_upload_file(remote_file)

    def run_upload_file(self, remote_file):
        print('remote_file')
        print(remote_file)
        with ssh_utils.sftp_client(**self.ssh_config) as client:
            ssh_utils.upload_file(client, self.file_a, remote_file)
            # check file in remote
            assert ssh_utils.exists(client, remote_file)

    def test_upload_dir(self):
        remote_dir = self.upload_dir()
        with ssh_utils.sftp_client(**self.ssh_config) as client:
            remote_dir_path = Path(remote_dir)
            assert ssh_utils.exists(client, remote_dir)
            remote_destination_dir_path = remote_dir_path / self.data_dir.name
            print("remote_destination_dir_path")
            print(remote_destination_dir_path)
            assert ssh_utils.exists(client, remote_destination_dir_path.as_posix())
            assert ssh_utils.exists(client, (remote_destination_dir_path / "a.txt").as_posix())
            assert ssh_utils.exists(client, (remote_destination_dir_path / "empty_dir").as_posix())
            assert ssh_utils.exists(client, (remote_destination_dir_path / "sub_dir").as_posix())
            assert ssh_utils.exists(client, (remote_destination_dir_path / "sub_dir" / "b.txt").as_posix())

