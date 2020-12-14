# -*- coding:utf-8 -*-
import argparse

from hypernets.dispatchers.predict.grpc.predict_service import serve


def main():
    parser = argparse.ArgumentParser('start predict server.')
    parser.add_argument('--port', '-port',
                        type=int, default=8030,
                        help='tcp port of the predict server')
    args, argv = parser.parse_known_args()

    server, _ = serve(f'0.0.0.0:{args.port}', ' '.join(argv))
    server.wait_for_termination()


if __name__ == '__main__':
    try:
        main()
        print('done')
    except KeyboardInterrupt as e:
        print(e)
