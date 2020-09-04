from .proto import spec_pb2_grpc, spec_pb2
import grpc


class SearchDriverClient(object):
    _instances = {}

    class TrailItemWrapper(object):
        def __init__(self, trail_no, space_file_path):
            super(SearchDriverClient.TrailItemWrapper, self).__init__()
            self.trail_no = trail_no
            self.space_file_path = space_file_path

    class RpcCodeWrapper(object):
        def __init__(self, code):
            super(SearchDriverClient.RpcCodeWrapper, self).__init__()
            self.__dict__['code'] = code

    @classmethod
    def instance(cls, server):
        inst = cls._instances.get(server)
        if inst is None:
            inst = cls(server)
            cls._instances[server] = inst

        return inst

    def __init__(self, server):
        super(SearchDriverClient, self).__init__()
        self.channel = grpc.insecure_channel(server)
        self.stub = spec_pb2_grpc.SearchDriverStub(self.channel)

    def __del__(self):
        self.channel.close()

    def next(self, executor_id):
        req = spec_pb2.NextRequest(executor_id=executor_id)
        res = self.stub.next(req)
        return SearchDriverClient.TrailItemWrapper(
            trail_no=res.trail_no, space_file_path=res.space_file_path)

    def report(self, trail_no, reward, code=0):
        req = spec_pb2.TrailStatus(code=code, trail_no=trail_no, reward=reward)
        res = self.stub.report(req)
        return SearchDriverClient.RpcCodeWrapper(code=res.code)
