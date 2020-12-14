import queue
import sys

import grpc

from hypernets.dispatchers.process.grpc.proto import proc_pb2_grpc
from hypernets.dispatchers.process.grpc.proto.proc_pb2 import DataChunk, ProcessRequest, DownloadRequest
from hypernets.utils import logging

logger = logging.get_logger(__name__)


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

        result = {'running': True}
        ack_queue = queue.Queue()

        if encoding:
            handles = {DataChunk.OUT: lambda x: stdout.write(x.decode(encoding)),
                       DataChunk.DATA: lambda x: stdout.write(x.decode(encoding)),
                       DataChunk.ERR: lambda x: stderr.write(x.decode(encoding)),
                       DataChunk.END: lambda x: result.update({'code': x, 'running': False}),
                       }
        else:
            handles = {DataChunk.OUT: lambda x: stdout.write(x),
                       DataChunk.DATA: lambda x: stdout.write(x),
                       DataChunk.ERR: lambda x: stderr.write(x),
                       DataChunk.END: lambda x: result.update({'code': x, 'running': False}),
                       }

        def fire_request():
            request = ProcessRequest(args=args,
                                     program=program if program else b'',
                                     buffer_size=buffer_size,
                                     encoding=encoding if encoding else b'',
                                     cwd=cwd if cwd else b'')
            yield request

            dummy = ProcessRequest()
            while result['running']:
                if ack_queue.get():
                    yield dummy

        try:
            response = self.stub.run(fire_request())
            for chunk in response:
                fn = handles.get(chunk.kind, None)
                if fn:
                    fn(chunk.data)
                else:
                    logger.warning(f'unexpected chunk kind: {chunk.kind}')
                ack = chunk.kind != DataChunk.END
                ack_queue.put(ack)
            if 'code' in result.keys():
                try:
                    return int(result['code'])
                except ValueError:
                    return 90
            else:
                return 0
        except grpc.RpcError as e:
            try:
                msg = f'RpcError {e.__class__.__name__}: {e.code()}'
                logger.error(msg)
            except Exception:
                import traceback
                msg = f'[GRPC {self.server}] {e.__class__.__name__}:\n'
                logger.error(msg + traceback.format_exc())
            return 99

    def download(self, remote_file_path, local_file_object, buffer_size=-1):
        encoding = getattr(local_file_object, 'encoding', None)

        def dump_with_encoding(data):
            local_file_object.write(data.decode(encoding))

        if encoding:
            fn = dump_with_encoding
        else:
            fn = local_file_object.write

        errs = []
        try:
            request = DownloadRequest(peer='hi',
                                      path=remote_file_path,
                                      encoding=encoding if encoding else '',
                                      buffer_size=buffer_size)
            response = self.stub.download(request)
            for chunk in response:
                if chunk.kind == DataChunk.DATA:
                    fn(chunk.data)
                elif chunk.kind == DataChunk.END:
                    pass  #
                else:
                    errs.append(f'[{chunk.kind}] ' + chunk.data.decode(encoding))

            if errs:
                raise Exception('\n'.join(errs))
        except grpc.RpcError as e:
            try:
                msg = f'RpcError {e.__class__.__name__}: {e.code()}'
                logger.error(msg)
            except Exception:
                import traceback
                msg = f'[GRPC {self.server}] {e.__class__.__name__}:\n'
                logger.error(msg + traceback.format_exc())

            raise Exception(msg)
