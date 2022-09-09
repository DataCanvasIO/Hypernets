import os
import io
import time
import subprocess
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

from paramiko import SFTPClient

from hypernets.hyperctl import consts
from hypernets.hyperctl.utils import http_portal
from hypernets.hyperctl.batch import _ShellJob
from hypernets.utils import logging as hyn_logging
from hypernets.utils import ssh_utils
import shutil


logger = hyn_logging.get_logger(__name__)


class NoResourceException(Exception):
    pass


class ShellExecutor:

    def __init__(self, job: _ShellJob, api_server_portal, environments=None):
        self.job = job
        self.api_server_portal = api_server_portal
        self.environments = environments

    def run(self, *, independent_tmp=True):
        pass

    def post(self):
        pass

    def prepare(self):
        pass

    def close(self):
        pass

    def status(self):
        raise NotImplemented

    def _make_run_shell_content(self, independent_tmp):

        template_vars = {
            consts.KEY_ENV_JOB_DATA_DIR: self.job.data_dir_path.as_posix(),
            consts.KEY_ENV_JOB_NAME: self.job.name,
            consts.KEY_ENV_JOB_WORKING_DIR: self.job.working_dir,  # default value
            consts.KEY_TEMPLATE_COMMAND: self.job.batch.job_command,
            consts.KEY_ENV_SERVER_PORTAL: self.api_server_portal
        }

        run_shell = str(consts.RUN_SH_TEMPLATE)
        for k, v in template_vars.items():
            run_shell = run_shell.replace(f"#{k}#", v)

        export_custom_envs_command = ""
        if self.environments is not None:
            for k, v in self.environments.items():
                export_command = f"export {k}={v}\n"
                export_custom_envs_command = export_custom_envs_command + export_command

        export_tmp_command = ""
        if independent_tmp:
            independent_tmp_path = (Path(tempfile.gettempdir()) / consts.BATCH_TEMP / self.job.name).as_posix()
            export_tmp_command = f"export TMP={independent_tmp_path}"

        run_shell = run_shell.replace(f"#{consts.P_HOST_ENV}", export_custom_envs_command)\
            .replace(f"#{consts.P_TMP_ENV}", export_tmp_command)

        return run_shell

    def _write_run_shell_script(self, file_path, independent_tmp):
        file_path = Path(file_path)
        # check parent
        if not file_path.parent.exists():
            os.makedirs(Path(file_path.parent), exist_ok=True)
        # write file
        shell_content = self._make_run_shell_content(independent_tmp)
        logger.debug(f'write shell script for job {self.job.name} to {file_path} ')
        with open(file_path, 'w', newline='\n') as f:
            f.write(shell_content)


class LocalShellExecutor(ShellExecutor):

    def __init__(self, *args, **kwargs):
        super(LocalShellExecutor, self).__init__(*args, **kwargs)
        self._process = None

    def prepare_assets(self):
        if len(self.job.assets) == 0:
            return
        else:
            if not self.job.resources_path.exists():
                os.makedirs(self.job.resources_path, exist_ok=True)

        for asset in self.job.assets:
            asset_path = Path(asset).absolute()
            asset_file = asset_path.as_posix()
            if not asset_path.exists():
                logger.warning(f"local dir {asset_path} not exists, skip to copy")
                continue
            if asset_path.is_dir():
                shutil.copytree(asset_file, self.job.resources_path.as_posix())
            else:
                shutil.copyfile(asset_file, (self.job.resources_path / asset_path.name).as_posix())

    def run(self, *, independent_tmp=True):
        # prepare data dir
        job_data_dir = Path(self.job.data_dir_path)

        if not job_data_dir.exists():
            os.makedirs(job_data_dir, exist_ok=True)

        self.prepare_assets()

        # create shell file & write to local
        self._write_run_shell_script(self.job.run_file, independent_tmp)

        command = f'sh {self.job.run_file}'

        self._process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        logger.debug(f'executed command {command} , pid is {self._process.pid}')

    def status(self):
        if self._process is None:
            return _ShellJob.STATUS_INIT
        process_ret_code = self._process.poll()
        if process_ret_code is None:
            return _ShellJob.STATUS_RUNNING
        if process_ret_code == 0:
            return _ShellJob.STATUS_SUCCEED
        return _ShellJob.STATUS_FAILED

    def kill(self):
        if self._process is not None:
            try:
                self._process.kill()
                self._process.terminate()
                logger.info(f"kill pid {self._process.pid}")
            except Exception as e:
                logger.exception("kill job exception", e)
            process_ret_code = self._process.poll()
            while process_ret_code is None:
                time.sleep(1)
                process_ret_code = self._process.poll()
            logger.info(f"pid exit with code {process_ret_code}")

        else:
            logger.warning(f"current executor is not running or closed ")

    def close(self):
        if self._process is not None:
            self._process.terminate()
        else:
            logger.warning(f"current executor is not running or closed ")


class SSHRemoteMachine:

    def __init__(self, connection, environments=None):

        # Note: requires unix server, does not support windows now
        self.connection = connection
        self.environments = environments

        self._usage = (0, 0, 0)

    def alloc(self, cpu, ram, gpu):
        cpu_count()
        if self._usage == (0, 0, 0):
            self._usage = (-1, -1, -1)
            return True
        else:
            return False

    def test_connection(self):
        client = None
        try:
            # check connections
            client = ssh_utils.create_ssh_client(**self.connection)
            assert client
            logger.info(f"connect to host {self.hostname} successfully")
        except Exception as e:
            logger.exception("test connection failed")
        finally:
            if client is not None:
                client.close()

    @staticmethod
    def total_resources():
        return cpu_count(), 0, 0

    def release(self, released_usage):
        self._usage = (0, 0, 0)

    @property
    def usage(self):
        return self._usage

    @property
    def hostname(self):
        return self.connection['hostname']


class RemoteShellExecutor(ShellExecutor):
    def __init__(self, job: _ShellJob, api_server_portal, machine: SSHRemoteMachine):
        super(RemoteShellExecutor, self).__init__(job, api_server_portal, environments=machine.environments)
        self.machine = machine

        self._command_ssh_client = None
        self._remote_process = None

    @property
    def connections(self):
        return self.machine.connection

    def prepare_assets(self, sftp_client):
        if len(self.job.assets) == 0:
            return

        for asset in self.job.assets:
            asset_path = Path(asset).absolute()
            asset_file = asset_path.as_posix()
            if not asset_path.exists():
                logger.warning(f"local dir {asset_path} not exists, skip to upload")
                continue
            if asset_path.is_dir():
                ssh_utils.upload_dir(sftp_client, asset_file, self.job.resources_path.as_posix())
            else:
                ssh_utils.upload_file(sftp_client, asset_file, (self.job.resources_path / asset_path.name).as_posix())

    def run(self, *,  independent_tmp=True):
        data_dir = self.job.data_dir_path.as_posix()
        logger.debug(f"create remote data dir {data_dir}")
        ssh_conn = None
        sftp_client = None
        try:
            ssh_conn = ssh_utils.create_ssh_client(**self.connections)

            sftp_client = ssh_conn.open_sftp()
            # with ssh_utils.sftp_client(**self.connections) as sftp_client:
            logger.debug(f"create remote job output dir {data_dir} ")
            ssh_utils.makedirs(sftp_client, data_dir)

            logger.debug(f"prepare to upload assets to {self.machine.hostname}")
            self.prepare_assets(sftp_client)  # share connection

            # copy file to remote
            # ssh_utils.upload_file(sftp_client, run_file, self.job.run_file)
            run_file_content = self._make_run_shell_content(independent_tmp).encode('utf-8')
            ssh_utils.upload_file_obj(sftp_client, io.BytesIO(run_file_content), self.job.run_file)
            logger.debug(f'upload run file to {self.job.run_file}')

            if sftp_client is not None:
                sftp_client.close()

            # execute command in async
            # self._command_ssh_client = ssh_utils.create_ssh_client(**self.connections)
            self._command_ssh_client = ssh_conn

            command = f'sh {self.job.run_file}'
            logger.debug(f'execute command {command}')
            self._remote_process = self._command_ssh_client.exec_command(command, get_pty=True)
        except Exception as e:
            if ssh_conn is not None:
                ssh_conn.close()
            if sftp_client is not None:
                sftp_client.close()
            raise e
        finally:
            pass

    def status(self):
        if self._remote_process is not None:
            stdout = self._remote_process[1]
            if stdout.channel.exit_status_ready():
                ret_code = stdout.channel.recv_exit_status()
                return _ShellJob.STATUS_SUCCEED if ret_code == 0 else _ShellJob.STATUS_FAILED
            else:
                return _ShellJob.STATUS_RUNNING
        else:
            return _ShellJob.STATUS_INIT

    def finished(self):
        return self.status() in _ShellJob.FINAL_STATUS

    def kill(self):
        self.close()

    def close(self):
        if self._command_ssh_client is not None:
            self._command_ssh_client.close()
        else:
            logger.warning(f"current executor is not running or closed ")


class ExecutorManager:

    def __init__(self, api_server_portal):
        self.api_server_portal = api_server_portal

        self._waiting_queue = []

    def allocated_executors(self):
        raise NotImplemented

    def waiting_executors(self):
        return self._waiting_queue

    def alloc_executor(self, job):
        raise NotImplemented

    def kill_executor(self, executor):
        raise NotImplemented

    def release_executor(self, executor):
        raise NotImplemented

    def get_executor(self, job):
        for e in self.allocated_executors():
            if e.job == job:
                return e
        return None

    def prepare(self):
        pass

    # def to_config(self):
    #     raise NotImplemented


class RemoteSSHExecutorManager(ExecutorManager):

    def __init__(self, api_server_portal, machines: List[SSHRemoteMachine]):
        super(RemoteSSHExecutorManager, self).__init__(api_server_portal)
        self.machines = machines

        self._executors_map = {}

    def allocated_executors(self):
        return self._executors_map.keys()

    def alloc_executor(self, job):
        for machine in self.machines:
            if machine.usage == (0, 0, 0):
                ret = machine.alloc(-1, -1, -1)  # lock resource
                if ret:
                    hostname = machine.hostname
                    logger.info(f'allocated resource for {job.name},'
                                f' data dir at: {hostname}{job.data_dir_path}')
                    executor = RemoteShellExecutor(job, self.api_server_portal, machine)

                    job.set_ext({
                        'backend_type': 'remote',
                        'hostname': hostname
                    })
                    # DOT NOT push anything to `_executors_map` or `_waiting_queue` if exception
                    self._executors_map[executor] = machine
                    self._waiting_queue.append(executor)  # put it to queue if ready
                    return executor
        raise NoResourceException

    def kill_executor(self, executor):
        self.release_executor(executor)

    def release_executor(self, executor):
        executor.close()  # close connection

        machine = self._executors_map.get(executor)
        logger.debug(f"release resource of machine {machine.hostname} ")

        machine.release(executor.job.resource)
        self._waiting_queue.remove(executor)


    def prepare(self):
        for machine in self.machines:
            machine.test_connection()


class LocalExecutorManager(ExecutorManager):

    def __init__(self, api_server_portal, environments=None):
        super(LocalExecutorManager, self).__init__(api_server_portal)
        self.environments = environments

        self._allocated_executors = []
        self._is_busy = False

    def allocated_executors(self):
        return self._allocated_executors

    def alloc_executor(self, job):
        if not self._is_busy:
            executor = LocalShellExecutor(job, self.api_server_portal, environments=self.environments)
            self._allocated_executors.append(executor)
            self._waiting_queue.append(executor)
            self._is_busy = True
            logger.info(f'allocated resource job {job.name}, data dir at: localhost{job.data_dir_path}')
            return executor
        raise NoResourceException

    def release_executor(self, executor):
        self._is_busy = False
        self._waiting_queue.remove(executor)
        executor.close()

    def kill_executor(self, executor: LocalShellExecutor):
        self.release_executor(executor)
        executor.kill()


def create_executor_manager(backend_conf, server_host, server_port):  # instance factory
    backend_conf = backend_conf if backend_conf is not None else {}
    server_portal = http_portal(server_host, server_port)
    backend_type = backend_conf.get('type', 'local')

    if backend_type == 'remote':
        machines = [SSHRemoteMachine(**_) for _ in backend_conf['machines']]
        # check remote host setting, only warning for remote backend
        if consts.HOST_LOCALHOST == server_host:
            logger.warning("recommended that set IP address that can be accessed in remote machines, "
                           "but now it's \"localhost\", and the task executed on the remote machines "
                           "may fail because it can't get information from the api server server,"
                           " you can set it in `server.host` ")
        return RemoteSSHExecutorManager(server_portal, machines)
    elif backend_type == 'local':
        return LocalExecutorManager(server_portal, environments=backend_conf.get('environments'))
    else:
        raise ValueError(f"unknown backend {backend_type}")
