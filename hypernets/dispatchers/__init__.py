# -*- coding:utf-8 -*-
import os
from hypernets.dispatchers.config import DispatchConfig
import argparse


def get_dispatcher(hyper_model, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', default='standalone',
                        choices=['driver', 'executor', 'standalone'],
                        help='process role, one of driver/executor/standalone')
    parser.add_argument('--driver',
                        help='address and port of the driver process, in format: "<hostname>:<port>"')
    args = parser.parse_args()

    if hyper_model.searcher.parallelizable:
        role = args.role
        driver_address = args.driver
        if role == 'driver':
            from .parallel_dispatcher import ParallelDriverDispatcher
            return ParallelDriverDispatcher(driver_address)
        elif role == 'executor':
            from .parallel_dispatcher import ParallelExecutorDispatcher
            return ParallelExecutorDispatcher(driver_address)

    return default_dispatcher()


def default_dispatcher():
    from .in_process_dispatcher import InProcessDispatcher
    return InProcessDispatcher()


def ssh_run():
    from hypernets.dispatchers.ssh.ssh_cluster import SshCluster
    parser = argparse.ArgumentParser('run HyperNets in ssh cluster.')
    parser.add_argument('--driver', '-driver',
                        help='address of the driver node')
    parser.add_argument('--driver-port', '-driver-port',
                        type=int, default=8001,
                        help='grpc port of the driver node, the executors will connect to this port, default 8081')
    parser.add_argument('--executors', '-executors',
                        required=True,
                        help='addresses of the executor nodes')
    parser.add_argument('--log-dir', '-log-dir',
                        default='logs',
                        help='local directory to store log files')
    parser.add_argument('commands',
                        nargs='+',
                        help='command to run')
    args = parser.parse_args()
    # conf = DispatchConfig.load_config()

    cluster = SshCluster(args.driver, args.driver_port,
                         args.executors.split(','),
                         args.log_dir,
                         *args.commands)
    cluster.start()


if __name__ == '__main__':
    ssh_run()

    print('done')
