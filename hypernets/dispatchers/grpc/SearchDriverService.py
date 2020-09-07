import os
import pickle
import queue
import sys
import time

from .proto import spec_pb2_grpc, spec_pb2


class SearchDriverObserver(object):
    def on_next(self, trail_no, space_sample):
        pass

    def on_report(self, trail_no, space_sample, success, reward, elapsed, message):
        pass


class SearchDriverService(spec_pb2_grpc.SearchDriverServicer):
    queued_pool = queue.Queue()
    reported_pool = queue.Queue()
    all_items = {}  # space_id -> SearchItem

    class SearchItem(object):
        def __init__(self, trail_no, space_file, space_sample):
            space_id = space_sample.space_id
            assert space_id not in SearchDriverService.all_items.keys()

            super(SearchDriverService.SearchItem, self).__init__()
            self.trail_no = trail_no
            self.space_file = space_file
            self.space_sample = space_sample
            self.space_id = space_sample.space_id
            self.queued_at = time.time()

            self.reward = float('nan')
            self.code = -1
            self.message = ''

            SearchDriverService.all_items[space_id] = self

        def __str__(self):
            return f'{self.__dict__}'

    def __init__(self, spaces_dir, on_next, on_report):
        super(SearchDriverService, self).__init__()

        os.makedirs(spaces_dir, exist_ok=True)

        self.spaces_dir = spaces_dir
        self.on_next = on_next
        self.on_report = on_report

    def beat(self, request, context):
        executor_id = request.id
        # print(f'[beat] from {executor_id}')
        return spec_pb2.RpcCode(code=0)

    def next(self, request, context):
        try:
            executor_id = request.id
            item = self.queued_pool.get(False)
            item.start_at = time.time()

            if self.on_next:
                try:
                    self.on_next(item)
                except Exception as e:
                    print(e, file=sys.stderr)

            print(f'[dispatch] [{executor_id}], trail_no={item.trail_no}, space_id:{item.space_id}')
            return spec_pb2.TrailItem(space_id=item.space_id, space_file_path=item.space_file)
        except queue.Empty:
            return spec_pb2.TrailItem(space_id='', space_file_path='')

    def report(self, request, context):
        executor_id = request.id
        space_id = request.space_id

        assert space_id in self.all_items.keys()

        item = self.all_items[space_id]
        item.reward = request.reward
        item.code = request.code.code
        item.message = request.message

        item.reported_at = time.time()
        self.reported_pool.put(item)

        if self.on_report:
            try:
                self.on_report(item)
            except Exception as e:
                print(e, file=sys.stderr)

        print(
            f'[report] [{executor_id}], trail_no={item.trail_no}, space_id={space_id}, reward={item.reward}, code={item.code}')

        return spec_pb2.RpcCode(code=0)

    def add(self, trail_no, space_sample):
        space_file = f'{self.spaces_dir}/space-{trail_no}.pkl'
        with open(space_file, 'wb') as f:
            pickle.dump(space_sample, f)
        item = SearchDriverService.SearchItem(trail_no, space_file, space_sample)
        print(f'[push] trail_no={item.trail_no}, space_id={item.space_id}, space_file={space_file}')
        self.queued_pool.put(item)

    def running_size(self):
        # print(f'total:{len(self.all_items.keys())}, reported:{self.reported_pool.qsize()}')

        return len(self.all_items.keys()) - self.reported_pool.qsize()


def serve(addr, spaces_dir, on_next=None, on_report=None):
    import grpc
    from concurrent import futures

    search_service = SearchDriverService(spaces_dir, on_next, on_report)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    spec_pb2_grpc.add_SearchDriverServicer_to_server(search_service, server)

    server.add_insecure_port(addr)
    server.start()

    return server, search_service
