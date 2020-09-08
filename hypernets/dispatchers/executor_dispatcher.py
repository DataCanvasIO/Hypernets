# -*- coding:utf-8 -*-
import os
import signal
import sys
import time
from threading import Thread

from grpc import RpcError

from ..core.dispatcher import Dispatcher


# from ..core.trial import *


class ExecutorDispatcher(Dispatcher):
    def __init__(self, driver_address):
        super(ExecutorDispatcher, self).__init__()
        self.driver_address = driver_address
        self.executor_id = os.environ.get("ID", None)
        if self.executor_id is None:
            import socket
            hostname = socket.gethostname()
            self.executor_id = f'executor-{hostname}'

    def dispatch(self, hyper_model, X, y, X_val, y_val, max_trails, dataset_id, trail_store,
                 **fit_kwargs):
        from ..dispatchers.grpc.SearchDriverClient import SearchDriverClient
        import pickle

        print(f'connect to driver {self.driver_address}', end='')
        client = SearchDriverClient.instance(self.driver_address, self.executor_id)
        self.wait_server(client)
        self.start_beat(client)

        trail_no = 1
        while True:
            try:
                item = client.next()
            except RpcError as e:
                msg = f'{e.__class__.__name__}: {e}'
                print(msg, 'exit.')
                break
            if len(item.space_id) == 0:
                print(f'Not found trail, wait and continue')
                time.sleep(1)
                continue

            print(f'new trail: space_id={item.space_id}, sample_file={item.space_file_path}')
            try:
                with open(item.space_file_path, 'rb') as f:
                    space_sample = pickle.load(f)

                for callback in hyper_model.callbacks:
                    # callback.on_build_estimator(hyper_model, space_sample, estimator, trail_no)
                    callback.on_trail_begin(hyper_model, space_sample, trail_no)

                trail = hyper_model._run_trial(space_sample, trail_no, X, y, X_val, y_val, **fit_kwargs)
                if trail.reward != 0:
                    improved = hyper_model.history.append(trail)
                    if improved:
                        hyper_model.best_model = hyper_model.last_model

                    for callback in hyper_model.callbacks:
                        callback.on_trail_end(hyper_model, space_sample, trail_no, trail.reward,
                                              improved, trail.elapsed)
                else:
                    for callback in hyper_model.callbacks:
                        callback.on_trail_error(hyper_model, space_sample, trail_no)

                print(f'----------------------------------------------------------------')
                print(f'space signatures: {hyper_model.history.get_space_signatures()}')
                print(f'----------------------------------------------------------------')
                if trail_store is not None:
                    trail_store.put(dataset_id, trail)

                code = 0 if trail.reward != 0 else -1
                client.report(item.space_id, code, trail.reward)
            except Exception as e:
                msg = f'{e.__class__.__name__}: {e}'
                print(msg, file=sys.stderr)
                client.report(item.space_id, -1, 0.0, msg)
            finally:
                trail_no += 1

        return trail_no

    @staticmethod
    def wait_server(client):
        start_at = time.time()

        def fn():
            while True:
                try:
                    client.beat()
                    done_at = time.time()
                    print(' connected (%.3f seconds).' % (done_at - start_at,))
                    break
                except RpcError as e:
                    # print('failed to connect to driver', e, 'try again after 3 seconds')
                    print('.', end='', flush=True)
                    time.sleep(3.0)

        t = Thread(target=fn, name='grpc-connect-thread')
        t.start()
        t.join()

    @staticmethod
    def start_beat(client):
        def fn():
            try:
                while True:
                    client.beat()
                    # print('-' * 20, 'beat ok')
                    time.sleep(3.0)
            except RpcError as e:
                print('beat error', e, 'terminating process...', file=sys.stderr)
                os.kill(os.getpid(), signal.SIGTERM)

        t = Thread(target=fn, name='grpc-beat-thread')
        t.daemon = True
        t.start()

        return t
