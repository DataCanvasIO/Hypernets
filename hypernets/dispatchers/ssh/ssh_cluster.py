# -*- coding:utf-8 -*-

from .ssh_process import SshProcess
from .local_process import LocalProcess
import os
import time


def get_ip_for(peer_address):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect((peer_address, 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


class SshCluster(object):
    def __init__(self, driver, driver_port, executors, log_dir, *args, **kwargs):
        super(SshCluster, self).__init__()
        assert isinstance(executors, (list, tuple)) and len(executors) > 0
        assert driver_port

        self.driver = driver
        self.driver_port = driver_port
        self.executors = executors
        self.log_dir = log_dir
        self.args = args
        self.kwargs = kwargs

    def start(self):
        os.makedirs(self.log_dir, exist_ok=True)
        tag = time.strftime('%Y%m%d%H%M%S')

        def to_driver_process():
            cmd = self._driver_cmd
            ssh_host = self.driver
            if ssh_host:
                out_file = f'{self.log_dir}/{tag}-driver-{ssh_host}.out'
                err_file = f'{self.log_dir}/{tag}-driver-{ssh_host}.err'
                return SshProcess(ssh_host, cmd, None, out_file, err_file)
            else:
                out_file = f'{self.log_dir}/{tag}-driver-localhost.out'
                err_file = f'{self.log_dir}/{tag}-driver-localhost.err'
                return LocalProcess(cmd, None, out_file, err_file)

        def to_executor_process(index):
            ssh_host = self.executors[index]
            cmd = self._executor_cmd
            out_file = f'{self.log_dir}/{tag}-executor-{index}-{ssh_host}.out'
            err_file = f'{self.log_dir}/{tag}-executor-{index}-{ssh_host}.err'
            if ssh_host == 'localhost' or ssh_host.startswith('127.0.'):
                return LocalProcess(cmd, None, out_file, err_file)
            else:
                return SshProcess(ssh_host, cmd, None, out_file, err_file)

        driver_process = to_driver_process()
        executor_processes = [to_executor_process(i) for i in range(0, len(self.executors))]
        all_processes = [driver_process, ] + executor_processes
        for p in all_processes: p.start()
        for p in all_processes: p.join()

    @property
    def _driver_cmd(self):
        cmds = ' '.join(self.args)
        driver = self.driver if self.driver else '0.0.0.0'
        ssh_cmd = f'python {cmds} --driver={driver}:{self.driver_port} --role=driver'
        return ssh_cmd

    @property
    def _executor_cmd(self):
        cmds = ' '.join(self.args)
        driver = self.driver if self.driver else get_ip_for(self.executors[0])
        # ssh_cmd = f'echo python --driver={driver}:{self.driver_port} --role=executor {cmds}'
        ssh_cmd = f'python {cmds} --driver={driver}:{self.driver_port} --role=executor'
        return ssh_cmd
