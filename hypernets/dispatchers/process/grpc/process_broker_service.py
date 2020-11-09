import queue
import subprocess
import time
from threading import Thread

from grpc import RpcError

from hypernets.dispatchers.process.grpc.proto import proc_pb2_grpc
from hypernets.dispatchers.process.grpc.proto.proc_pb2 import DataChunk
from hypernets.utils import logging

logger = logging.get_logger(__name__)


class ProcessBrokerService(proc_pb2_grpc.ProcessBrokerServicer):
    def __init__(self):
        super(ProcessBrokerService, self).__init__()

    @staticmethod
    def _read_data(f, q, buffer_size, encoding, data_kind):

        try:
            data = f.read(buffer_size)
            while data and len(data) > 0:
                if encoding:
                    chunk = DataChunk(kind=data_kind, data=data.encode(encoding))
                else:
                    chunk = DataChunk(kind=data_kind, data=data)
                q.put(chunk)
                data = f.read(buffer_size)
        except ValueError as e:
            logger.error(e)

    def run(self, request_iterator, context):
        it = iter(request_iterator)
        request = next(it)

        program = request.program
        args = request.args
        cwd = request.cwd
        buffer_size = request.buffer_size
        encoding = request.encoding
        if encoding is None or len(encoding) == 0:
            encoding = None

        with subprocess.Popen(args, buffer_size,
                              program if len(program) > 0 else None,
                              cwd=cwd if len(cwd) > 0 else None,
                              stdin=None,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              encoding=encoding,
                              shell=False) as p:
            pid = p.pid
            start_at = time.time()
            peer = context.peer()
            if logger.is_info_enabled():
                logger.info(f'[{pid}] started, peer: {peer}, cmd:' + ' '.join(args), )

            data_queue = queue.Queue()
            t_out = Thread(target=self._read_data,
                           args=(p.stdout, data_queue, buffer_size, encoding, DataChunk.OUT))
            t_err = Thread(target=self._read_data,
                           args=(p.stderr, data_queue, buffer_size, encoding, DataChunk.ERR))
            t_out.start()
            t_err.start()

            # report pid to client
            yield DataChunk(kind=DataChunk.ERR, data=f'pid: {pid}\n'.encode())

            try:
                while next(it):
                    chunk = None
                    while context.is_active() and \
                            (t_out.is_alive() or t_err.is_alive() or not data_queue.empty()):
                        try:
                            chunk = data_queue.get(False)
                            yield chunk
                            break
                        except queue.Empty:
                            time.sleep(0.1)
                    if not context.is_active():
                        p.kill()
                        code = 'killed (peer shutdown)'
                        break
                    elif chunk is None:  # process exit and no more output
                        code = p.poll()
                        yield DataChunk(kind=DataChunk.END, data=str(code).encode())
                        # break
            except StopIteration as e:
                pass
            except RpcError as e:
                logger.error(e)
                code = 'rpc error'
            except Exception:
                import traceback
                traceback.print_exc()
                code = 'exception'

        if logger.is_info_enabled():
            logger.info('[%s] done with code %s, elapsed %.3f seconds.'
                        % (pid, code, time.time() - start_at))

    def download(self, request, context):
        try:
            peer = request.peer
            path = request.path
            encoding = request.encoding
            buffer_size = request.buffer_size
            if buffer_size is None or buffer_size <= 0:
                buffer_size = 4096

            # check peer here

            start_at = time.time()
            total = 0
            if encoding:
                with open(path, 'r', encoding=encoding) as f:
                    data = f.read(buffer_size)
                    while data and len(data) > 0:
                        if not context.is_active():
                            break
                        encoded_data = data.encode(encoding)
                        chunk = DataChunk(kind=DataChunk.DATA, data=encoded_data)
                        total += len(encoded_data)
                        yield chunk
                        data = f.read(buffer_size)
            else:
                with open(path, 'rb') as f:
                    data = f.read(buffer_size)
                    while data and len(data) > 0:
                        if not context.is_active():
                            break
                        chunk = DataChunk(kind=DataChunk.DATA, data=data)
                        total += len(data)
                        yield chunk
                        data = f.read(buffer_size)

            if not context.is_active():
                if logger.is_info_enabled():
                    logger.info('download %s broke (peer shutdown), %s bytes sent, elapsed %.3f seconds' %
                                (path, total, time.time() - start_at))
            else:
                yield DataChunk(kind=DataChunk.END, data=b'')
                if logger.is_info_enabled():
                    logger.info('download %s (%s bytes) in %.3f seconds, encoding=%s' %
                                (path, total, time.time() - start_at, encoding))
        except Exception as e:
            import traceback
            import sys
            msg = f'{e.__class__.__name__}:\n'
            msg += traceback.format_exc()
            logger.error(msg)
            yield DataChunk(kind=DataChunk.EXCEPTION, data=msg.encode())


def serve(addr, max_workers=10):
    import grpc
    from concurrent import futures

    if logger.is_info_enabled():
        logger.info(f'start broker at {addr}')
    service = ProcessBrokerService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    proc_pb2_grpc.add_ProcessBrokerServicer_to_server(service, server)

    server.add_insecure_port(addr)
    server.start()

    return server, service
