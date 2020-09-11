# -*- coding:utf-8 -*-
import argparse
from .ssh.ssh_cluster import SshCluster


def main():
    parser = argparse.ArgumentParser('run HyperNets experiment in cluster.')
    parser.add_argument('--experiment', '-experiment',
                        default=None,
                        help='experiment id, default current timestamp')
    parser.add_argument('--driver-broker', '-driver-broker',
                        help='address of the driver broker' +
                             ', eg: grpc://<hostname>:<broker_port> to use grpc broker'
                             + ', or <hostname> to use ssh broker')
    parser.add_argument('--driver-port', '-driver-port',
                        type=int, default=8001,
                        help='tcp port of the driver'
                             + ', the executors will connect to this port with grpc'
                             + ', default 8001')
    parser.add_argument('--executor-brokers', '-executor-brokers',
                        required=True,
                        help='addresses of the executor nodes, separated by comma'
                             + ' eg: "grpc://<executor1_hostname>:<broker_port>' +
                             ',grpc://<executor2_hostname>:<broker_port>"')
    parser.add_argument('--spaces-dir', '-spaces-dir',
                        default='tmp',
                        help='driver directory to store space files, default "tmp"')
    parser.add_argument('--logs-dir', '-logs-dir',
                        default='logs',
                        help='local directory to store log files')
    args, argv = parser.parse_known_args()

    cluster = SshCluster(args.experiment,
                         args.driver_broker, args.driver_port,
                         args.executor_brokers.split(','),
                         args.spaces_dir,
                         args.logs_dir,
                         *argv)
    cluster.start()


if __name__ == '__main__':
    main()

    print('done')
