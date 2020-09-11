# -*- coding:utf-8 -*-

import os
import re
import time

from .grpc_process import GrpcProcess
from .local_process import LocalProcess
from .ssh_process import SshProcess


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


class AddressParser(object):

    def __init__(self, addr):
        super(AddressParser, self).__init__()

        pattern = r'(([a-zA-Z0-9]+)://)?([-a-zA-Z0-9+&@#/%?=~_.]+[-a-zA-Z0-9+&@#/%?=~_])(:(\d+))?'
        m = re.match(pattern, addr)
        if m is None:
            msg = f'invalid address: {addr}'
            raise Exception(msg)

        protocol = m.group(2)
        host = m.group(3)
        port = m.group(5)

        if protocol is None:
            protocol = 'ssh'
        if protocol not in {'ssh', 'grpc'}:
            raise Exception('not supported protocol: ' + protocol)

        if port is None:
            if protocol == 'ssh':
                port = '22'
            else:
                port = '8010'

        self.protocol = protocol
        self.host = host
        self.port = int(port)

    @property
    def is_ssh(self):
        return self.protocol == 'ssh'

    @property
    def is_grpc(self):
        return self.protocol == 'grpc'

    @property
    def is_localhost(self):
        return self.host == 'localhost' or self.host.startswith('127.0.')

    def __str__(self):
        return f'{self.protocol}://{self.host}:{self.port}'


class SshCluster(object):

    def __init__(self, experiment,
                 driver, driver_port, executors,
                 spaces_dir, logs_dir,
                 *args, **kwargs):
        super(SshCluster, self).__init__()
        assert isinstance(executors, (list, tuple)) and len(executors) > 0
        assert driver_port

        self.experiment = experiment if experiment else time.strftime('%Y%m%d%H%M%S')
        self.driver = driver
        self.driver_port = driver_port
        self.executors = executors
        self.spaces_dir = spaces_dir
        self.logs_dir = logs_dir
        self.args = args
        self.kwargs = kwargs

    def start(self):
        tag = self.experiment
        os.makedirs(f'{self.logs_dir}/{tag}', exist_ok=True)

        def log_for(address):
            return f'{address.host}-{address.port}'

        def to_driver_process():
            cmd = self._driver_cmd
            a = AddressParser(self.driver if self.driver else 'localhost')
            if a.is_localhost:
                return LocalProcess(cmd, None, None, None)
            elif a.is_grpc:
                grpc_host = f'{a.host}:{a.port}'
                return GrpcProcess(grpc_host, cmd, None, None, None)
            else:
                return SshProcess(a.host, a.port, cmd, None, None, None)

        def to_executor_process(index):
            a = AddressParser(self.executors[index])
            cmd = self._executor_cmd

            out_file = f'{self.logs_dir}/{tag}/executor-{index}-{log_for(a)}.out'
            err_file = f'{self.logs_dir}/{tag}/executor-{index}-{log_for(a)}.err'

            if a.is_localhost:
                return LocalProcess(cmd, None, out_file, err_file)
            elif a.is_grpc:
                grpc_host = f'{a.host}:{a.port}'
                return GrpcProcess(grpc_host, cmd, None, out_file, err_file)
            else:
                return SshProcess(a.host, a.port, cmd, None, out_file, err_file)

        driver_process = to_driver_process()
        executor_processes = [to_executor_process(i) for i in range(0, len(self.executors))]
        all_processes = [driver_process, ] + executor_processes

        [p.start() for p in all_processes]
        [p.join() for p in all_processes]

        # codes = [p.exitcode for p in all_processes]
        # print('process exit code:', codes)

        return driver_process.exitcode

    @property
    def _driver_cmd(self):
        cmds = ' '.join(self.args)
        driver = self.driver if self.driver else '0.0.0.0'
        driver_host = AddressParser(driver).host
        spaces_dir = f'{self.spaces_dir}/{self.experiment}'

        ssh_cmd = f'{cmds} --driver {driver_host}:{self.driver_port} --role driver --spaces-dir {spaces_dir}'
        return ssh_cmd

    @property
    def _executor_cmd(self):
        cmds = ' '.join(self.args)
        if self.driver:
            a = AddressParser(self.driver)
            driver_host = a.host
        else:  # at localhost
            a = AddressParser(self.executors[0])
            driver_host = get_ip_for(a.host)

        ssh_cmd = f'{cmds} --driver {driver_host}:{self.driver_port} --role executor'
        return ssh_cmd
