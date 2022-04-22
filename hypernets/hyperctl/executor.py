import os
import time
import subprocess
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

from paramiko import SFTPClient

from hypernets.hyperctl import consts
from hypernets.hyperctl.utils import http_portal
from hypernets.hyperctl.batch import ShellJob
from hypernets.utils import logging as hyn_logging
from hypernets.utils import ssh_utils


logger = hyn_logging.get_logger(__name__)


class NoResourceException(Exception):
    pass


class ShellExecutor:

    def __init__(self, job: ShellJob, api_server_portal):
        self.job = job
        self.api_server_portal = api_server_portal

    def run(self):
        pass

    def post(self):
        pass

    def prepare(self):
        pass

    def close(self):
        pass

    def status(self):
        raise NotImplemented

    def _make_run_shell_content(self):
        # default http://localhost:8060
        vars = {
            consts.KEY_ENV_JOB_DATA_DIR: self.job.job_data_dir,
            consts.KEY_ENV_JOB_NAME: self.job.name,
            consts.KEY_ENV_JOB_WORKING_DIR: self.job.working_dir,  # default value
            consts.KEY_TEMPLATE_COMMAND: self.job.command,
            consts.KEY_ENV_SERVER_PORTAL: self.api_server_portal,
        }

        run_shell = str(consts.RUN_SH_TEMPLATE)
        for k, v in vars.items():
            run_shell = run_shell.replace(f"#{k}#", v)

        return run_shell

    def _write_run_shell_script(self, file_path):
        file_path = Path(file_path)
        # check parent
        if not file_path.parent.exists():
            os.makedirs(Path(file_path.parent), exist_ok=True)
        # write file
        shell_content = self._make_run_shell_content()
        logger.debug(f'write shell script for job {self.job.name} to {file_path} ')
        with open(file_path, 'w', newline='\n') as f:
            f.write(shell_content)


class RemoteShellExecutor(ShellExecutor):
    def __init__(self, job: ShellJob, api_server_portal, connection):
        super(RemoteShellExecutor, self).__init__(job, api_server_portal)
        self.connections = connection

        self._command_ssh_client = None
        self._remote_process = None

    def prepare_assets(self):
        # upload resource to working dir
        with ssh_utils.sftp_client(**self.connections) as sftp_client:
            logger.debug(f"create remote job output dir {output_dir} ")
            ssh_utils.makedirs(sftp_client, output_dir)
            for asset in self.job.assets:
                # 将文件一个个上传
                sftp_client.upload_dir(asset)




                Path(asset).absolute().as_posix()  # upload to
            pass

    def run(self):
        # create remote data dir
        output_dir = Path(self.job.output_dir).as_posix()
        with ssh_utils.sftp_client(**self.connections) as sftp_client:
            logger.debug(f"create remote job output dir {output_dir} ")
            ssh_utils.makedirs(sftp_client, output_dir)

        # create run shell file
        fd_run_file, run_file = tempfile.mkstemp(prefix=f'hyperctl_run_{self.job.name}_', suffix='.sh')
        os.close(fd_run_file)

        self._write_run_shell_script(run_file)

        # copy file to remote
        with ssh_utils.sftp_client(**self.connections) as sftp_client:
            logger.debug(f'upload {run_file} to {self.job.run_file_path}')
            sftp_client: SFTPClient = sftp_client
            ssh_utils.upload_file(sftp_client, run_file, self.job.run_file_path)

        # execute command in async
        self._command_ssh_client = ssh_utils.create_ssh_client(**self.connections)
        command = f'sh {self.job.run_file_path}'
        logger.debug(f'execute command {command}')
        self._remote_process = self._command_ssh_client.exec_command(command, get_pty=True)

    def status(self):
        if self._remote_process is not None:
            stdout = self._remote_process[1]
            if stdout.channel.exit_status_ready():
                ret_code = stdout.channel.recv_exit_status()
                return ShellJob.STATUS_SUCCEED if ret_code == 0 else ShellJob.STATUS_FAILED
            else:
                return ShellJob.STATUS_RUNNING
        else:
            return ShellJob.STATUS_INIT

    def finished(self):
        return self.status() in ShellJob.FINAL_STATUS

    def kill(self):
        self.close()

    def close(self):
        if self._command_ssh_client is not None:
            self._command_ssh_client.close()
        else:
            logger.warning(f"current executor is not running or closed ")


class LocalShellExecutor(ShellExecutor):

    def __init__(self, job: ShellJob, api_server_portal):
        super(LocalShellExecutor, self).__init__(job, api_server_portal)
        self._process = None

    def run(self):
        # prepare data dir
        job_data_dir = Path(self.job.job_data_dir)
        if not job_data_dir.exists():
            os.makedirs(job_data_dir, exist_ok=True)

        # create shell file & write to local
        self._write_run_shell_script(self.job.run_file_path)

        command = f'sh {self.job.run_file_path}'

        self._process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        logger.debug(f'executed command {command} , pid is {self._process.pid}')

    def status(self):
        if self._process is None:
            return ShellJob.STATUS_INIT
        process_ret_code = self._process.poll()
        if process_ret_code is None:
            return ShellJob.STATUS_RUNNING
        if process_ret_code == 0:
            return ShellJob.STATUS_SUCCEED
        return ShellJob.STATUS_FAILED

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

    def __init__(self, connection):

        # Note: requires unix server, does not support windows now
        self.connection = connection

        self._usage = (0, 0, 0)

    def alloc(self, cpu, ram, gpu):
        cpu_count()
        if self._usage == (0, 0, 0):
            self._usage = (-1, -1, -1)
            return True
        else:
            return False

    def test_connection(self):
        # check connections
        with ssh_utils.ssh_client(**self.connection) as client:
            assert client
            logger.info(f"test connection to host {self.connection['hostname']} successfully")

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

    def __init__(self, machines: List[SSHRemoteMachine], api_server_portal):
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
                    logger.debug(f'allocated resource on {machine.hostname} for job {job.name} ')
                    executor = RemoteShellExecutor(job, self.api_server_portal, machine.connection)
                    self._executors_map[executor] = machine
                    self._waiting_queue.append(executor)
                    return executor
        raise NoResourceException

    def kill_executor(self, executor):
        self.release_executor(executor)

    def release_executor(self, executor):
        machine = self._executors_map.get(executor)
        logger.debug(f"release resource of machine {machine.hostname} ")
        machine.release(executor.job.resource)
        self._waiting_queue.remove(executor)
        executor.close()

    def prepare(self):
        for machine in self.machines:
            machine.test_connection()


class LocalExecutorManager(ExecutorManager):

    def __init__(self, api_server_portal):
        super(LocalExecutorManager, self).__init__(api_server_portal)
        self._allocated_executors = []
        self._is_busy = False

    def allocated_executors(self):
        return self._allocated_executors

    def alloc_executor(self, job):
        if not self._is_busy:
            executor = LocalShellExecutor(job, self.api_server_portal)
            self._allocated_executors.append(executor)
            self._waiting_queue.append(executor)
            self._is_busy = True
            return executor
        raise NoResourceException

    def release_executor(self, executor):
        self._is_busy = False
        self._waiting_queue.remove(executor)
        executor.close()

    def kill_executor(self, executor: LocalShellExecutor):
        self.release_executor(executor)
        executor.kill()


def create_executor_manager(backend_type, backend_conf, server_host, server_port):  # instance factory
    server_portal = http_portal(server_host, server_port)
    if backend_type == 'remote':
        remote_backend_config = backend_conf
        machines = [SSHRemoteMachine(_) for _ in remote_backend_config['machines']]
        # check remote host setting, only warning for remote backend
        if consts.HOST_LOCALHOST == server_host:
            logger.warning("recommended that set IP address that can be accessed in remote machines, "
                           "but now it's \"localhost\", and the task executed on the remote machines "
                           "may fail because it can't get information from the api server server,"
                           " you can set it in `server.host` ")
        return RemoteSSHExecutorManager(machines, server_portal)
    elif backend_type == 'local':
        return LocalExecutorManager(server_portal)
    else:
        raise ValueError(f"unknown backend {backend_type}")
