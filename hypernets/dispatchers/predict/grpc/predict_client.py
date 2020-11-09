import grpc

from hypernets.dispatchers.predict.grpc.proto import predict_pb2_grpc
from hypernets.dispatchers.predict.grpc.proto.predict_pb2 import PredictRequest
from hypernets.utils import logging

logger = logging.get_logger(__name__)


class PredictClient(object):

    def __init__(self, server):
        super(PredictClient, self).__init__()
        self.channel = grpc.insecure_channel(server)
        self.stub = predict_pb2_grpc.PredictServiceStub(self.channel)

        self.server = server
        self._closed = False

    def __del__(self):
        self.close()

    def close(self):
        if not self._closed:
            self.channel.close()

    def predict(self, data_file, result_file):
        try:
            request = PredictRequest(data_file=data_file, result_file=result_file)
            response = self.stub.predict(request)
            code = response.code
            return code
        except Exception as e:
            import traceback
            msg = f'[Predict {self.server}] {e.__class__.__name__}:\n'
            logger.error(msg + traceback.format_exc())

            return 98 if isinstance(e, grpc.RpcError) else 99
