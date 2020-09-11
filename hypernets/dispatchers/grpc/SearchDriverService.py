import os
import pickle
import queue
import sys
import time
from threading import Thread

from .proto import spec_pb2_grpc, spec_pb2
from ...utils.common import config

executor_beat_timeout = int(config('executor_beat_timeout', '60'))  # seconds


class SearchDriverObserver(object):
    def on_next(self, trail_no, space_sample):
        pass

    def on_report(self, trail_no, space_sample, success, reward, elapsed, message):
        pass


class SearchItem(object):
    def __init__(self, trail_no, space_file, space_sample):
        super(SearchItem, self).__init__()

        self.trail_no = trail_no
        self.space_file = space_file
        self.space_sample = space_sample
        self.space_id = space_sample.space_id
        self.queue_at = time.time()

        self.reward = float('nan')
        self.code = -1
        self.message = ''

    def __str__(self):
        return f'{self.__dict__}'


class ExecutorItem(object):
    def __init__(self, executor_id):
        super(ExecutorItem, self).__init__()

        now = time.time()
        self.executor_id = executor_id
        self.start_at = now
        self.beat_at = now

    def beat(self):
        self.beat_at = time.time()

    @property
    def alive(self):
        now = time.time()
        return abs(now - self.beat_at) < executor_beat_timeout


class SearchDriverService(spec_pb2_grpc.SearchDriverServicer):
    def __init__(self, spaces_dir, on_next, on_report, on_summary):
        super(SearchDriverService, self).__init__()

        os.makedirs(spaces_dir, exist_ok=True)

        self.spaces_dir = spaces_dir
        self.on_next = on_next
        self.on_report = on_report
        self.on_summary = on_summary

        self.executors = {}  # executor_id -> ExecutorItem

        self.queued_pool = queue.Queue()  # SearchItem

        self.running_items = {}  # space_id -> SearchItem
        self.reported_items = {}  # space_id -> SearchItem
        self.all_items = {}  # space_id -> SearchItem

        self.status_thread = DriverStatusThread(self)
        self.status_thread.start()

    def __del__(self):
        self.status_thread.stop()

    def register(self, request, context):
        executor_id = request.id

        index = 0
        while f'{executor_id}-{index}' in self.executors.keys():
            index += 1

        registered_id = f'{executor_id}-{index}'

        item = ExecutorItem(registered_id)
        self.executors[registered_id] = item

        print(f'[register] executor={registered_id}')

        return spec_pb2.ExecutorId(id=registered_id)

    def beat(self, request, context):
        executor_id = request.id

        item = self.executors.get(executor_id)
        if item is None:
            return spec_pb2.RpcCode(code=-1)
        else:
            item.beat()
            return spec_pb2.RpcCode(code=0)

    def next(self, request, context):
        assert request.id in self.executors.keys()

        try:
            item = self.queued_pool.get(False)
        except queue.Empty:
            return spec_pb2.TrailItem(space_id='', space_file_path='')

        executor_id = request.id
        item.start_at = time.time()
        item.executor_id = executor_id

        if self.on_next:
            try:
                self.on_next(item)
            except Exception as e:
                print(e, file=sys.stderr)
        self.running_items[item.space_id] = item

        print(f'[dispatch] [{executor_id}], trail_no={item.trail_no}, space_id:{item.space_id}')
        return spec_pb2.TrailItem(space_id=item.space_id, space_file_path=item.space_file)

    def report(self, request, context):
        assert request.id in self.executors.keys()

        executor_id = request.id
        space_id = request.space_id
        assert space_id in self.all_items.keys()

        item = self.all_items[space_id]
        detail = f'trail_no={item.trail_no}, space_id={space_id}' \
                 + f', reward={request.reward}, code={request.code.code}' \
                 + f', message={request.message}'

        if space_id not in self.running_items.keys():
            msg = f'[ignored-not running-report] [{executor_id}] {detail}'
            print(msg, file=sys.stderr)
        elif executor_id != item.executor_id:
            msg = f'[ignored-invalid executor-report] [{executor_id}] {detail}' \
                  + f', expect executor={item.executor_id}'
            print(msg, file=sys.stderr)
        else:
            item.reward = request.reward
            item.code = request.code.code
            item.message = request.message
            item.report_at = time.time()

            self.running_items.pop(space_id)
            self.reported_items[space_id] = item

            if self.on_report:
                try:
                    self.on_report(item)
                except Exception as e:
                    print(e, file=sys.stderr)

            print(f'[report] [{executor_id}] {detail}')

        return spec_pb2.RpcCode(code=0)

    def add(self, trail_no, space_sample):
        space_id = space_sample.space_id
        assert space_id not in self.all_items.keys()

        space_file = f'{self.spaces_dir}/space-{trail_no}.pkl'
        with open(space_file, 'wb') as f:
            pickle.dump(space_sample, f)

        item = SearchItem(trail_no, space_file, space_sample)
        print(f'[push] trail_no={item.trail_no}, space_id={item.space_id}, space_file={space_file}')

        self.queued_pool.put(item)
        self.all_items[space_id] = item

    def readd(self, item):
        space_id = item.space_id
        space_file = item.space_file
        assert space_id in self.all_items.keys()

        detail = f'trail_no={item.trail_no}, executor_id={item.executor_id}' \
                 + f', space_id={item.space_id}, space_file={space_file}'
        print(f'[re-push] {detail}')

        if space_id in self.running_items.keys():
            print(f'[remove running] {detail}')
            self.running_items.pop(space_id)

        if space_id in self.reported_items.keys():
            print(f'[remove reported] {detail}')
            self.reported_items.pop(space_id)

        self.queued_pool.put(item)

    def running_size(self):
        return len(self.all_items) - len(self.reported_items)

    def queue_size(self):
        return self.queued_pool.qsize()


class DriverStatusThread(Thread):
    def __init__(self, service):
        super(DriverStatusThread, self).__init__()

        self.service = service
        self.daemon = True
        self.running = False
        self.summary_interval = float(config('summary_interval', '60'))

    def run(self) -> None:
        self.running = True

        summary_at = time.time()
        while self.running:
            try:
                self.check_health()

                now = time.time()
                if self.report_summary and self.summary_interval > 0 and summary_at + self.summary_interval < now:
                    self.report_summary()
                    summary_at = now

                time.sleep(1)
            except Exception as e:
                print(e, file=sys.stderr)

        print('SearchDriverService shutdown')

    def stop(self):
        self.running = False

    def check_health(self):
        service = self.service
        dead_executors = filter(lambda x: not x.alive, service.executors.values())
        dead_executor_id_set = set(map(lambda x: x.executor_id, dead_executors))
        if len(dead_executor_id_set) == 0:
            return

        dead_items = filter(lambda x: x.executor_id in dead_executor_id_set,
                            service.running_items.values())
        for item in list(dead_items).copy():
            service.readd(item)

    def report_summary(self):
        service = self.service
        total = len(service.all_items)
        running = len(service.running_items)
        reported = len(service.reported_items)
        queued = total - running - reported

        msg = f'[summary] queued={queued}, running={running}, reported={reported}, total={total}'
        if running > 0:
            detail = [lambda x: f'trail {x.trail_no}:{x.executor_id}', service.running_items]
            msg += '\nrunning: ' + ','.join(detail)

        if service.on_summary:
            service_summary = service.on_summary()
            if service_summary:
                msg += '\n>>> ' + service_summary

        print(msg)


def serve(addr, spaces_dir, on_next=None, on_report=None, on_summary=None):
    import grpc
    from concurrent import futures

    search_service = SearchDriverService(spaces_dir, on_next, on_report, on_summary)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    spec_pb2_grpc.add_SearchDriverServicer_to_server(search_service, server)

    server.add_insecure_port(addr)
    server.start()

    return server, search_service
