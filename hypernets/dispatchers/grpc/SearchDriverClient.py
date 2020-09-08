from .proto import spec_pb2_grpc, spec_pb2
import grpc
import sys


class SearchDriverClient(object):
    _instances = {}
    callbacks = []

    class TrailItemWrapper(object):
        def __init__(self, space_id, space_file_path):
            super(SearchDriverClient.TrailItemWrapper, self).__init__()
            self.space_id = space_id
            self.space_file_path = space_file_path

    class RpcCodeWrapper(object):
        def __init__(self, code):
            super(SearchDriverClient.RpcCodeWrapper, self).__init__()
            self.__dict__['code'] = code

    @classmethod
    def instance(cls, server, executor_id):
        key = f'{server}/{executor_id}'
        inst = cls._instances.get(key)
        if inst is None:
            inst = cls(server, executor_id)
            cls._instances[key] = inst

        return inst

    def __init__(self, server, executor_id):
        super(SearchDriverClient, self).__init__()
        self.channel = grpc.insecure_channel(server)
        self.stub = spec_pb2_grpc.SearchDriverStub(self.channel)

        self.server = server
        self.executor_id = executor_id

    def __del__(self):
        self.channel.close()

    def beat(self):
        req = spec_pb2.ExecutorId(id=self.executor_id)
        res = self.stub.beat(req)
        return SearchDriverClient.RpcCodeWrapper(code=res.code)

    def next(self):
        req = spec_pb2.ExecutorId(id=self.executor_id)
        # print(f'call next with: {req}')
        res = self.stub.next(req)
        return SearchDriverClient.TrailItemWrapper(
            space_id=res.space_id, space_file_path=res.space_file_path)

    def report(self, space_id, code, reward, message=''):
        req = spec_pb2.TrailReport(id=self.executor_id,
                                   space_id=space_id,
                                   code=spec_pb2.RpcCode(code=code),
                                   reward=reward,
                                   message=message)
        # print(f'call report with: {req}')
        res = self.stub.report(req)
        return SearchDriverClient.RpcCodeWrapper(code=res.code)
