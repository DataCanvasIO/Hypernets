import sys

import grpc

from .proto import proc_pb2_grpc
from .proto.proc_pb2 import DataChunk, ProcessRequest


class ProcessBrokerClient(object):

    def __init__(self, server):
        super(ProcessBrokerClient, self).__init__()
        self.channel = grpc.insecure_channel(server)
        self.stub = proc_pb2_grpc.ProcessBrokerStub(self.channel)

        self.server = server

    def __del__(self):
        self.channel.close()

    def run(self, args, program=None, buffer_size=-1, cwd=None, stdout=None, stderr=None):
        if stdout is None:
            stdout = sys.stdout
        if stderr is None:
            stdout = sys.stderr

        encoding = getattr(stdout, 'encoding', None)

        request = ProcessRequest(args=args,
                                 program=program if program else b'',
                                 buffer_size=buffer_size,
                                 encoding=encoding if encoding else b'',
                                 cwd=cwd if cwd else b'')

        result = {}
        if encoding:
            handles = {DataChunk.OUT: lambda x: stdout.write(x.decode(encoding)),
                       DataChunk.ERR: lambda x: stderr.write(x.decode(encoding)),
                       DataChunk.END: lambda x: result.update({'code': x}),
                       }
        else:
            handles = {DataChunk.OUT: lambda x: stdout.write(x),
                       DataChunk.ERR: lambda x: stderr.write(x),
                       DataChunk.END: lambda x: result.update({'code': x}),
                       }

        try:
            response = self.stub.run(request)
            for chunk in response:
                fn = handles.get(chunk.kind, None)
                if fn:
                    fn(chunk.data)
                else:
                    print(f'unexpected chunk kind: {chunk.kind}', file=sys.stderr)

            if 'code' in result.keys():
                try:
                    return int(result['code'])
                except ValueError:
                    return 90
            else:
                return 0
        except grpc.RpcError as e:
            import traceback
            msg = f'[GRPC {self.server}] {e.__class__.__name__}:\n'
            print(msg + traceback.format_exc(), file=sys.stderr)

            return 99
