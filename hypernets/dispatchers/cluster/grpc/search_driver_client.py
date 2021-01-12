import queue
import time

import grpc

from hypernets.utils import logging
from .proto import spec_pb2_grpc
from .proto.spec_pb2 import SearchRequest, SearchResponse, PingMessage

logger = logging.get_logger(__name__)


class SearchDriverClient(object):
    _instances = {}
    callbacks = []

    class TrialItemWrapper(object):
        def __init__(self, code, search_id, trial_no,
                     space_id, space_file, model_file):
            super(SearchDriverClient.TrialItemWrapper, self).__init__()

            self.code = code
            self.search_id = search_id
            try:
                self.trial_no = int(trial_no)
            except ValueError:
                self.trial_no = None
            self.space_id = space_id
            self.space_file = space_file
            self.model_file = model_file

            self.success = False
            self.reward = None
            self.message = None

        def is_ok(self):
            return self.code == SearchResponse.OK

        def is_waiting(self):
            return self.code == SearchResponse.WAITING

        def is_finished(self):
            return self.code == SearchResponse.FINISHED

        def to_request(self):
            msg = SearchRequest(search_id=self.search_id,
                                space_id=self.space_id,
                                trial_no=str(self.trial_no) if self.trial_no is not None else '',
                                success=self.success,
                                reward=self.reward if self.reward is not None else 0.0,
                                message=self.message if self.message else '')
            return msg

        @classmethod
        def from_response(cls, msg):
            item = cls(code=msg.code,
                       search_id=msg.search_id,
                       trial_no=msg.trial_no,
                       space_id=msg.space_id,
                       space_file=msg.space_file,
                       model_file=msg.model_file)
            return item

    def __init__(self, server, search_id):
        super(SearchDriverClient, self).__init__()
        self.channel = grpc.insecure_channel(server)
        self.stub = spec_pb2_grpc.SearchDriverStub(self.channel)

        self.server = server
        self.search_id = search_id

    def close(self):
        self.channel.close()

    def ping(self, wait=False, message=None):

        def default_message():
            try:
                import socket
                hostname = socket.gethostname()
                return f'from-{hostname}'
            except Exception:
                return 'ping'

        def do_ping():
            req = PingMessage(message=message if message else default_message())
            res = self.stub.ping(req)
            return res.message

        if wait:
            start_at = time.time()
            result = ''
            while True:
                try:
                    result = do_ping()
                    done_at = time.time()
                    print(' connected with %.3f seconds.' % (done_at - start_at))
                    break
                except grpc.RpcError:
                    print('.', end='', flush=True)
                    time.sleep(1)

            return result
        else:
            return do_ping()

    def search(self, search_id):

        ack_queue = queue.Queue()
        running = True

        def fire_request():
            # fire first msg,
            msg = SearchRequest(search_id=search_id,
                                trial_no='',
                                space_id='',
                                success=False,
                                reward=0.0,
                                message='')
            yield msg

            # fire response msg and get next trial
            while running:
                try:
                    r = ack_queue.get(block=False)
                    if r is None:
                        break
                    msg = r.to_request()
                    yield msg
                except queue.Empty:
                    time.sleep(0.1)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    break

        try:
            response = self.stub.search(fire_request())
            for res in response:
                item = SearchDriverClient.TrialItemWrapper.from_response(res)
                try:
                    result = yield item
                    ack_queue.put(result)
                except Exception as e:
                    msg = f'{e.__class__.__name__}: {e}'
                    logger.error(msg)
        except grpc.RpcError as e:
            import traceback
            trace_detail = traceback.format_exc()
            try:
                msg = f'RpcError {self.server} {e.__class__.__name__}: {e.code()}'
                logger.error(msg)
            except Exception:
                msg = f'RpcError {self.server} {e.__class__.__name__}:\n'
                logger.error(msg + trace_detail)
            yield None
        running = False
