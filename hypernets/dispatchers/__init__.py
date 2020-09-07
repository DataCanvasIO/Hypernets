# -*- coding:utf-8 -*-
import argparse


def get_dispatcher(hyper_model, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', '-role',
                        default='standalone', choices=['driver', 'executor', 'standalone'],
                        help='process role, one of driver/executor/standalone')
    parser.add_argument('--driver', '-driver',
                        help='address and port of the driver process, with format: "<hostname>:<port>"')
    parser.add_argument('--spaces-dir', '-spaces-dir',
                        default='spaces',
                        help='[driver only] director to store space sample file, default "spaces"')
    args, _ = parser.parse_known_args()

    if hyper_model.searcher.parallelizable:
        role = args.role
        driver_address = args.driver
        if role == 'driver':
            from .driver_dispatcher import DriverDispatcher
            return DriverDispatcher(driver_address, args.spaces_dir)
        elif role == 'executor':
            from .executor_dispatcher import ExecutorDispatcher
            return ExecutorDispatcher(driver_address)

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
    parser.add_argument('--experiment', '-experiment',
                        help='driver directory to store space files')
    parser.add_argument('--spaces-dir', '-spaces-dir',
                        default='spaces',
                        help='driver directory to store space files')
    parser.add_argument('--logs-dir', '-logs-dir',
                        default='logs',
                        help='local directory to store log files')
    args, argv = parser.parse_known_args()

    cluster = SshCluster(args.experiment,
                         args.driver, args.driver_port,
                         args.executors.split(','),
                         args.spaces_dir,
                         args.logs_dir,
                         *argv)
    cluster.start()


if __name__ == '__main__':
    ssh_run()

    print('done')
