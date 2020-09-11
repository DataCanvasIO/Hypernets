import queue
import subprocess
import time
from threading import Thread

from .proto import proc_pb2_grpc
from .proto.proc_pb2 import DataChunk


class ProcessBrokerService(proc_pb2_grpc.ProcessBrokerServicer):
    def __init__(self):
        super(ProcessBrokerService, self).__init__()

    @staticmethod
    def _read_data(f, q, buffer_size, encoding, data_kind):

        data = f.read(buffer_size)
        while data and len(data) > 0:
            if encoding:
                chunk = DataChunk(kind=data_kind, data=data.encode(encoding))
            else:
                chunk = DataChunk(kind=data_kind, data=data)
            # print(f'put: {data}')
            q.put(chunk)
            data = f.read(buffer_size)

    def run(self, request, context):
        # print(f'req: {request}')
        program = request.program
        args = request.args
        cwd = request.cwd
        buffer_size = request.buffer_size
        encoding = request.encoding
        if encoding is None or len(encoding) == 0:
            encoding = None
        # print(' '.join(args), program, cwd, buffer_size, encoding)

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
            print(f'[{pid}] started: ' + ' '.join(args))

            data_queue = queue.Queue()
            t_out = Thread(target=self._read_data,
                           args=(p.stdout, data_queue, buffer_size, encoding, DataChunk.OUT))
            t_err = Thread(target=self._read_data,
                           args=(p.stderr, data_queue, buffer_size, encoding, DataChunk.ERR))
            t_out.start()
            t_err.start()

            while t_out.is_alive() or t_err.is_alive() or not data_queue.empty():
                try:
                    chunk = data_queue.get(False)
                    # print(f'return: {chunk}')
                    yield chunk
                except queue.Empty:
                    time.sleep(0.1)

            code = p.poll()

            yield DataChunk(kind=DataChunk.END, data=str(code).encode())

        print('[%s] done with code %s, elapsed %.3f seconds.'
              % (pid, code, time.time() - start_at))


def serve(addr):
    import grpc
    from concurrent import futures

    print(f'start broker at {addr}')
    service = ProcessBrokerService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proc_pb2_grpc.add_ProcessBrokerServicer_to_server(service, server)

    server.add_insecure_port(addr)
    server.start()

    return server, service
