# -*- coding:utf-8 -*-
import argparse

from hypernets.dispatchers.process.grpc.process_broker_service import serve


def main():
    parser = argparse.ArgumentParser('run HyperNets process broker.')
    parser.add_argument('--host', '-host',
                        default='0.0.0.0',
                        help='broker hostname or ip address'
                             + ', default "0.0.0.0"')
    parser.add_argument('--port', '-port',
                        type=int, default=8010,
                        help='broker tcp port, default 8010')
    parser.add_argument('--workers', '-workers',
                        type=int, default=10,
                        help='max worker count, default 10')

    args = parser.parse_args()

    server, _ = serve(f'{args.host}:{args.port}', args.workers)
    server.wait_for_termination()


if __name__ == '__main__':
    try:
        main()
        print('done')
    except KeyboardInterrupt as e:
        print(e)
