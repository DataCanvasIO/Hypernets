from .proto import spec_pb2_grpc, spec_pb2
import queue
import pickle
import os
import time


class SearchDriverService(spec_pb2_grpc.SearchDriverServicer):
    queued_pool = queue.Queue()
    reported_pool = queue.Queue()
    all_items = {}  # trail_no -> SearchItem

    spaces_dir = 'spaces'
    os.makedirs(spaces_dir, exist_ok=True)

    class SearchItem(object):
        def __init__(self, trail_no, space_file, space_sample):
            assert trail_no not in SearchDriverService.all_items.keys()

            super(SearchDriverService.SearchItem, self).__init__()
            self.trail_no = trail_no
            self.space_file = space_file
            self.space_sample = space_sample
            self.queued_at = time.time()

            self.reward = float('nan')
            self.code = -1

            SearchDriverService.all_items[trail_no] = self

    def next(self, request, context):
        try:
            executor_id = request.executor_id
            item = self.queued_pool.get(False)
            item.start_at = time.time()

            print(f'[call next] executor_id:{executor_id}, trail_no:{item.trail_no}')
            return spec_pb2.TrailItem(trail_no=item.trail_no, space_file_path=item.space_file)
        except queue.Empty:
            return spec_pb2.TrailItem(trail_no=-1, space_file_path="")

    def report(self, request, context):
        code = request.code
        trail_no = request.trail_no
        reward = request.reward

        print(f'[call report] trail={trail_no}, remark={reward}, code={code}')
        assert trail_no in self.all_items.keys()

        item = self.all_items[trail_no]
        item.reward = reward
        item.code = code
        item.reported_at = time.time()

        self.reported_pool.put(item)

        return spec_pb2.RpcCode(code=0)

    def add(self, trail_no, space_sample):
        space_file = f'{self.spaces_dir}/space-{trail_no}.pkl'
        with open(space_file, 'wb') as f:
            pickle.dump(space_sample, f)
        item = SearchDriverService.SearchItem(trail_no, space_file, space_sample)
        self.queued_pool.put(item)

    def running_size(self):
        print(f'total:{len(self.all_items.keys())}, reported:{self.reported_pool.qsize()}')

        return len(self.all_items.keys()) - self.reported_pool.qsize()


def serve(addr):
    import grpc
    from concurrent import futures

    search_service = SearchDriverService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    spec_pb2_grpc.add_SearchDriverServicer_to_server(search_service, server)

    server.add_insecure_port(addr)
    server.start()

    return server, search_service
