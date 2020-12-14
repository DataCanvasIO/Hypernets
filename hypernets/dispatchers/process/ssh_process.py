# -*- coding:utf-8 -*-

import sys
from multiprocessing import Process, Value as PValue, current_process
from os.path import getsize
from threading import Thread

from paramiko import SSHClient, AutoAddPolicy

from hypernets.utils import logging
from hypernets.utils.common import Counter

logger = logging.get_logger(__name__)


class DumpFileThread(Thread):
    counter = Counter()

    def __init__(self, in_file_handle, out_file_handle, buf_size=16):
        super(DumpFileThread, self).__init__()
        assert in_file_handle and out_file_handle

        # self.name = f'{self.__class__.__name__}-{self.counter()}'
        self.name = f'{self.__class__.__name__}-{current_process().pid}-{self.counter()}'
        self.in_file_handle = in_file_handle
        self.out_file_handle = out_file_handle
        self.buf_size = buf_size

    def run(self):
        data = self.in_file_handle.read(self.buf_size)
        while data and len(data) > 0:
            self.out_file_handle.write(data)
            data = self.in_file_handle.read(self.buf_size)


class SshProcess(Process):
    def __init__(self, ssh_host, ssh_port, cmd, in_file, out_file, err_file, environment=None):
        super(SshProcess, self).__init__()
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.cmd = cmd
        self.in_file = in_file
        self.out_file = out_file
        self.err_file = err_file
        self.environment = environment
        self._exit_code = PValue('i', -1)

    def run(self):
        if logger.is_info_enabled():
            logger.info(f'[{self.name}] [SSH {self.ssh_host}]: {self.cmd}')
        try:
            code = self.ssh_run(self.ssh_host, self.ssh_port,
                                self.cmd,
                                self.in_file,
                                self.out_file,
                                self.err_file,
                                self.environment)
        except KeyboardInterrupt:
            code = 137

        if logger.is_info_enabled():
            logger.info(f'[{self.name}] [SSH {self.ssh_host}] {self.cmd} done with {code}')

        self._exit_code.value = code

    @staticmethod
    def ssh_run(ssh_host, ssh_port, cmd, in_file, out_file, err_file, environment):
        with SSHClient() as ssh:
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            ssh.connect(ssh_host, ssh_port)
            stdin, stdout, stderr = ssh.exec_command(cmd, bufsize=10, environment=environment)
            if in_file and getsize(in_file) > 0:
                with open(in_file, 'rb') as f:
                    data = f.read()
                    stdin.write(data)
            stdin.flush()

            channel = stdout.channel
            # channel.settimeout(0.1)

            if out_file and err_file:
                with open(out_file, 'wb', buffering=0)as o, open(err_file, 'wb', buffering=0) as e:
                    threads = [DumpFileThread(stdout, o), DumpFileThread(stderr, e)]
                    for p in threads: p.start()
                    for p in threads: p.join()
            else:
                threads = [DumpFileThread(stdout, sys.stdout), DumpFileThread(stderr, sys.stderr)]
                for p in threads: p.start()
                for p in threads: p.join()

            assert channel.exit_status_ready()
            code = channel.recv_exit_status()
        return code

    @property
    def exitcode(self):
        code = self._exit_code.value
        return code if code >= 0 else None
