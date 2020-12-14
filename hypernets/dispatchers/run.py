# -*- coding:utf-8 -*-
import argparse

from hypernets.dispatchers.cluster import Cluster


def main():
    parser = argparse.ArgumentParser('run HyperNets experiment in cluster.')
    parser.add_argument('--experiment', '-experiment',
                        default=None,
                        help='experiment id, default current timestamp')
    parser.add_argument('--driver-broker', '-driver-broker',
                        help='address of the driver broker'
                             ', eg: grpc://<hostname>:<broker_port> to use grpc process broker'
                             ', or just <hostname> to use ssh process')
    parser.add_argument('--driver-port', '-driver-port',
                        type=int, default=8001,
                        help='tcp port of the driver'
                             ', the executors will connect to this port with grpc'
                             ', default 8001')
    parser.add_argument('--executor-brokers', '-executor-brokers',
                        required=True,
                        help='addresses of the executor nodes, separated by comma. '
                             'eg: "grpc://<executor1_hostname>:<broker_port>,'
                             'grpc://<executor2_hostname>:<broker_port>"')
    parser.add_argument('--with-driver', '-with-driver',
                        type=int, default=1,
                        help='start driver progress or not, default 1')
    parser.add_argument('--spaces-dir', '-spaces-dir',
                        default='tmp',
                        help='driver directory to store space files, default "tmp"')
    parser.add_argument('--logs-dir', '-logs-dir',
                        default='logs',
                        help='local directory to store log files')
    parser.add_argument('--report-interval', '-report-interval',
                        type=int, default=60,
                        help='report cluster processes, default 60')
    args, argv = parser.parse_known_args()

    cluster = Cluster(args.experiment,
                      args.driver_broker,
                      args.driver_port,
                      args.with_driver,
                      args.executor_brokers.split(','),
                      args.spaces_dir,
                      args.logs_dir,
                      args.report_interval,
                      *argv)
    cluster.run()


if __name__ == '__main__':
    try:
        main()
        print('done')
    except KeyboardInterrupt as e:
        print('KeyboardInterrupt')
