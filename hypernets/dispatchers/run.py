# -*- coding:utf-8 -*-
import argparse
from .ssh.ssh_cluster import SshCluster


def main():
    parser = argparse.ArgumentParser('run HyperNets in ssh cluster.')
    parser.add_argument('--experiment', '-experiment',
                        default=None,
                        help='experiment id, default current timestamp')
    parser.add_argument('--driver', '-driver',
                        help='address of the driver node')
    parser.add_argument('--driver-port', '-driver-port',
                        type=int, default=8001,
                        help='tcp port of the driver, the executors will connect to this port with grpc, default 8081')
    parser.add_argument('--executors', '-executors',
                        required=True,
                        help='addresses of the executor nodes, separated by comma')
    parser.add_argument('--spaces-dir', '-spaces-dir',
                        default='tmp',
                        help='driver directory to store space files, default "tmp"')
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
    main()

    print('done')
