# -*- coding:utf-8 -*-

import os
import re
import time
from threading import Thread

from hypernets.dispatchers.process import GrpcProcess, LocalProcess, SshProcess
from hypernets.utils import logging

logger = logging.get_logger(__name__)


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


class Cluster(object):

    def __init__(self, experiment,
                 driver, driver_port, start_driver,
                 executors,
                 work_dir, logs_dir,
                 report_interval,
                 *args, **kwargs):
        super(Cluster, self).__init__()
        assert isinstance(executors, (list, tuple)) and len(executors) > 0
        assert driver_port
        assert isinstance(args, (list, tuple)) and len(args) > 0

        self.experiment = experiment if experiment else time.strftime('%Y%m%d%H%M%S')
        self.driver = driver
        self.driver_port = driver_port
        self.start_driver = start_driver
        self.executors = executors
        self.work_dir = work_dir
        self.logs_dir = logs_dir
        self.report_interval = report_interval
        self.args = args
        self.kwargs = kwargs

    def run(self):
        tag = self.experiment
        os.makedirs(f'{self.logs_dir}/{tag}', exist_ok=True)

        def log_for(address):
            return f'{address.host}-{address.port}'

        def to_driver_process():
            cmd = self._driver_cmd
            a = AddressParser(self.driver if self.driver else 'localhost')
            if a.is_localhost:
                p = LocalProcess(cmd, None, None, None)
            elif a.is_grpc:
                grpc_host = f'{a.host}:{a.port}'
                p = GrpcProcess(grpc_host, cmd, None, None, None)
                return p
            else:
                p = SshProcess(a.host, a.port, cmd, None, None, None)

            p.name = 'driver'
            return p

        def to_executor_process(index):
            a = AddressParser(self.executors[index])
            cmd = self._executor_cmd

            out_file = f'{self.logs_dir}/{tag}/executor-{index}-{log_for(a)}.out'
            err_file = f'{self.logs_dir}/{tag}/executor-{index}-{log_for(a)}.err'

            if a.is_localhost:
                p = LocalProcess(cmd, None, out_file, err_file)
            elif a.is_grpc:
                grpc_host = f'{a.host}:{a.port}'
                p = GrpcProcess(grpc_host, cmd, None, out_file, err_file)
            else:
                p = SshProcess(a.host, a.port, cmd, None, out_file, err_file)

            p.name = f'executor-{index}'

            return p

        executor_processes = [to_executor_process(i) for i in range(0, len(self.executors))]
        if self.start_driver:
            driver_process = to_driver_process()
            all_processes = [driver_process, ] + executor_processes
        else:
            all_processes = executor_processes

        [p.start() for p in all_processes]

        status_thread = None
        if self.report_interval and self.report_interval > 0:
            status_thread = ClusterStatusThread(all_processes, self.report_interval)
            status_thread.start()

        [p.join() for p in all_processes]

        if status_thread:
            status_thread.stop()
            status_thread.report()

    @property
    def _driver_cmd(self):
        cmds = ' '.join(self.args)
        driver = self.driver if self.driver else '0.0.0.0'
        driver_host = AddressParser(driver).host

        ssh_cmd = f'{cmds} --driver {driver_host}:{self.driver_port} --role driver'
        if self.experiment:
            ssh_cmd += f' --experiment {self.experiment}'
        if self.work_dir:
            ssh_cmd += f' --work-dir {self.work_dir}'

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


class ClusterStatusThread(Thread):
    def __init__(self, processes, interval=60):
        super(ClusterStatusThread, self).__init__()

        self.daemon = True

        self.processes = processes
        self.interval = interval
        self.running = False

    def run(self):
        assert not self.running

        self.running = True
        time.sleep(self.interval)

        while self.running:
            try:
                self.report()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(e)

    def stop(self):
        self.running = False

    def report(self):
        def summary(p):
            if p.is_alive():
                return f'[{p.pid}] {p.name}: running'
            else:
                return f'[{p.pid}] {p.name}: done with {p.exitcode}'

        if logger.is_info_enabled():
            msg = '\n'.join([summary(p) for p in self.processes])
            logger.info('Cluster status: >>>\n' + msg + '\n<<<')
