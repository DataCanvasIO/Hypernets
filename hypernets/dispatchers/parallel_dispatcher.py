# -*- coding:utf-8 -*-

import time

from ..core.callbacks import EarlyStoppingError
from ..core.dispatcher import Dispatcher
from ..core.trial import *


class ParallelDriverDispatcher(Dispatcher):
    def __init__(self, address):
        super(ParallelDriverDispatcher, self).__init__()
        self.address = address

    def dispatch(self, hyper_model, X, y, X_val, y_val, max_trails, dataset_id, trail_store,
                 **fit_kwargs):
        from .grpc.SearchDriverService import serve
        server, search_service = serve(self.address)
        print(f'serve at {self.address}')

        trail_no = 1
        retry_counter = 0
        while trail_no <= max_trails:
            space_sample = hyper_model.searcher.sample()
            if hyper_model.history.is_existed(space_sample):
                if retry_counter >= 1000:
                    print(f'Unable to take valid sample and exceed the retry limit 1000.')
                    break
                trail = hyper_model.history.get_trail(space_sample)
                for callback in hyper_model.callbacks:
                    callback.on_skip_trail(hyper_model, space_sample, trail_no, 'trail_exsited', trail.reward, False,
                                           trail.elapsed)
                retry_counter += 1
                continue
            # for testing
            # space_sample = self.searcher.space_fn()
            # trails = self.trail_store.get_all(dataset_id, space_sample1.signature)
            # space_sample.assign_by_vectors(trails[0].space_sample_vectors)
            # space_sample.space_id = space_sample1.space_id

            try:
                if trail_store is not None:
                    trail = trail_store.get(dataset_id, space_sample)
                    if trail is not None:
                        reward = trail.reward
                        elapsed = trail.elapsed
                        trail = Trail(space_sample, trail_no, reward, elapsed)
                        improved = hyper_model.history.append(trail)
                        if improved:
                            hyper_model.best_model = None
                        hyper_model.searcher.update_result(space_sample, reward)
                        for callback in hyper_model.callbacks:
                            callback.on_skip_trail(hyper_model, space_sample, trail_no, 'hit_trail_store', reward,
                                                   improved,
                                                   elapsed)
                        trail_no += 1
                        continue

                # trail = hyper_model._run_trial(space_sample, trail_no, X, y, X_val, y_val, **fit_kwargs)
                # print(f'----------------------------------------------------------------')
                # print(f'space signatures: {hyper_model.history.get_space_signatures()}')
                # print(f'----------------------------------------------------------------')
                # if trail_store is not None:
                #     trail_store.put(dataset_id, trail)
                search_service.add(trail_no, space_sample)
                print(f'push trail {trail_no} to queue')

            except EarlyStoppingError:
                break
                # TODO: early stopping
            except Exception as e:
                print(f'{">" * 20} Trail failed! {"<" * 20}')
                print(e)
                print('*' * 50)
            finally:
                trail_no += 1
                retry_counter = 0

        print("-" * 20)
        print("-- wait execute --")
        while search_service.running_size() > 0:
            # print(f"wait ... {search_service.running_size()} samples found.")
            time.sleep(3)

        print("-" * 20)
        print("-- collect result --")
        for item in search_service.all_items.values():
            elapsed = item.reported_at - item.start_at
            trail = Trail(item.space_sample, item.trail_no, item.reward, elapsed)
            print(f'trail:{trail.trail_no}, reward:{trail.reward}, elapsed:{trail.elapsed}')
            ####
            # fixme, remove following from _run_trial
            improved = hyper_model.history.append(trail)
            if improved:
                estimator = hyper_model._get_estimator(item.space_sample)
                hyper_model.best_model = estimator.model  # fixme, load model from disk?
            hyper_model.searcher.update_result(item.space_sample, item.reward)
            #####

            if trail_store is not None:
                trail_store.put(dataset_id, trail)
        print("-" * 20)

        return trail_no


class ParallelExecutorDispatcher(Dispatcher):
    def __init__(self, driver_address):
        super(ParallelExecutorDispatcher, self).__init__()
        self.driver_address = driver_address
        self.executor_id = os.environ.get("ID", "unknown")

    def dispatch(self, hyper_model, X, y, X_val, y_val, max_trails, dataset_id, trail_store,
                 **fit_kwargs):
        from ..dispatchers.grpc.SearchDriverClient import SearchDriverClient
        import pickle

        client = SearchDriverClient.instance(self.driver_address)
        while True:
            try:
                item = client.next(f"foo-{self.executor_id}")
            except Exception as e:
                print(e)
                print(f'No more tail, exit.')
                return

            if item.trail_no < 0:
                print(f'invalid trail:{item.trail_no}, wait and continue')
                time.sleep(1)
                continue

            print(f'new trail:{item.trail_no}, sample_file:{item.space_file_path}')

            trail_no = item.trail_no
            trail_reward = 0.0
            try:
                with open(item.space_file_path, 'rb') as f:
                    space_sample = pickle.load(f)

                trail = hyper_model._run_trial(space_sample, trail_no, X, y, X_val, y_val, **fit_kwargs)
                print(f'----------------------------------------------------------------')
                print(f'space signatures: {hyper_model.history.get_space_signatures()}')
                print(f'----------------------------------------------------------------')
                if trail_store is not None:
                    trail_store.put(dataset_id, trail)
                trail_reward = trail.reward
            except Exception as e:
                print(e)
                print(f'No more tail, exit.')

            client.report(trail_no, trail_reward)
