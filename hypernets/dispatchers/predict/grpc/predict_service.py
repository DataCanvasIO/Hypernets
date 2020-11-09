import time

from hypernets.dispatchers.predict.grpc.proto import predict_pb2_grpc
from hypernets.dispatchers.predict.grpc.proto.predict_pb2 import PredictResponse
from hypernets.dispatchers.process import LocalProcess
from hypernets.utils import logging

logger = logging.get_logger(__name__)


class PredictService(predict_pb2_grpc.PredictServiceServicer):
    def __init__(self, cmd):
        super(PredictService, self).__init__()
        assert cmd

        self.cmd = cmd

    def predict(self, request, context):
        data_file = request.data_file
        result_file = request.result_file

        start_at = time.time()

        if logger.is_info_enabled():
            print(f'predict {data_file} --> {result_file}', end='')

        cmd = f'{self.cmd} {data_file} {result_file}'
        p = LocalProcess(cmd, None, None, None)
        p.start()
        p.join()
        code = p.exitcode

        res = PredictResponse(data_file=data_file, result_file=result_file, code=code)

        done_at = time.time()
        if logger.is_info_enabled():
            print(' done, elapsed %.3f seconds.' % (done_at - start_at))
        return res


def serve(addr, cmd):
    import grpc
    from concurrent import futures

    if logger.is_info_enabled():
        logger.info(f'start predict service at {addr}')
    service = PredictService(cmd)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    predict_pb2_grpc.add_PredictServiceServicer_to_server(service, server)

    server.add_insecure_port(addr)
    server.start()

    return server, service
