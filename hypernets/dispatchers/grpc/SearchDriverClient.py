import sys
import time
from threading import Thread

import grpc

from .proto import spec_pb2_grpc, spec_pb2


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
        self.registered = False

    def __del__(self):
        self.channel.close()

    def register(self, wait=False):
        if self.registered:
            return self.executor_id

        def do_register():
            req = spec_pb2.ExecutorId(id=self.executor_id)
            res = self.stub.register(req)
            self.registered = True
            self.executor_id = res.id
            return self.executor_id

        if wait:
            start_at = time.time()
            while not self.registered:
                try:
                    do_register()
                    done_at = time.time()
                    print(' registered as %s with %.3f seconds.' % (self.executor_id, done_at - start_at))
                except grpc.RpcError:
                    print('.', end='', flush=True)
                    time.sleep(1)

            return self.executor_id
        else:
            return do_register()

    def beat(self):
        if not self.registered:
            self.register()

        req = spec_pb2.ExecutorId(id=self.executor_id)
        res = self.stub.beat(req)
        return SearchDriverClient.RpcCodeWrapper(code=res.code)

    def start_beat_thread(self):
        def fn():
            try:
                while True:
                    self.beat()
                    # print('-' * 20, 'beat ok')
                    time.sleep(3.0)
            except grpc.RpcError as e:
                msg = f'{e.__class__.__name__}: {e}'
                print('beat error', msg, 'stop beat', file=sys.stderr)

        t = Thread(target=fn, name='beat-thread')
        t.daemon = True
        t.start()

        return t

    def next(self):
        if not self.registered:
            self.register()

        req = spec_pb2.ExecutorId(id=self.executor_id)
        # print(f'call next with: {req}')
        res = self.stub.next(req)
        return SearchDriverClient.TrailItemWrapper(
            space_id=res.space_id, space_file_path=res.space_file_path)

    def report(self, space_id, code, reward, message=''):
        if not self.registered:
            self.register()

        req = spec_pb2.TrailReport(id=self.executor_id,
                                   space_id=space_id,
                                   code=spec_pb2.RpcCode(code=code),
                                   reward=reward,
                                   message=message)
        # print(f'call report with: {req}')
        res = self.stub.report(req)
        return SearchDriverClient.RpcCodeWrapper(code=res.code)
