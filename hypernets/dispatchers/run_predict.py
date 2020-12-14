# -*- coding:utf-8 -*-
import argparse

from hypernets.dispatchers.predict.predict_helper import PredictHelper


def main():
    parser = argparse.ArgumentParser('run predict.')
    parser.add_argument('--server', '-server',
                        default='127.0.0.1:8030',
                        help='predict server address, separated by comma')
    parser.add_argument('--chunk-size', '-chunk-size',
                        type=int, default=1000,
                        help='chunk line number')
    parser.add_argument('data_file',
                        help='data file path')
    parser.add_argument('result_file',
                        help='result file path')
    args = parser.parse_args()

    servers = list(filter(lambda s: len(s) > 0, args.server.split(',')))
    ph = PredictHelper(servers)
    ph.predict(args.data_file, args.result_file, args.chunk_size)


if __name__ == '__main__':
    try:
        main()
        print('done')
    except KeyboardInterrupt as e:
        print(e)
