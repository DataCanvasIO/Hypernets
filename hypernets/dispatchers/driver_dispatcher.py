# -*- coding:utf-8 -*-

import time

from ..core.callbacks import EarlyStoppingError
from ..core.dispatcher import Dispatcher
from ..core.trial import Trail


class DriverDispatcher(Dispatcher):
    def __init__(self, address, spaces_dir):
        super(DriverDispatcher, self).__init__()
        self.address = address
        self.spaces_dir = spaces_dir

    def dispatch(self, hyper_model, X, y, X_val, y_val, max_trails, dataset_id, trail_store,
                 **fit_kwargs):
        def on_next_space(item):
            for cb in hyper_model.callbacks:
                # cb.on_build_estimator(hyper_model, space_sample, estimator, trail_no)
                cb.on_trail_begin(hyper_model, item.space_sample, item.trail_no)

        def on_report_space(item):
            if item.code == 0:
                elapsed = item.report_at - item.start_at
                trail = Trail(item.space_sample, item.trail_no, item.reward, elapsed)
                # print(f'trail result:{trail}')

                improved = hyper_model.history.append(trail)
                if improved:
                    # estimator = hyper_model._get_estimator(item.space_sample)
                    # hyper_model.best_model = estimator.model  # fixme, load model from executor disk?
                    hyper_model.best_space = item.space_sample
                    print(f'>>>improved: reward={item.reward}, trail_no={item.trail_no}, space_id={item.space_id}')
                hyper_model.searcher.update_result(item.space_sample, item.reward)

                if trail_store is not None:
                    trail_store.put(dataset_id, trail)

                for cb in hyper_model.callbacks:
                    cb.on_trail_end(hyper_model, item.space_sample, item.trail_no, item.reward, improved, elapsed)
            else:
                for cb in hyper_model.callbacks:
                    cb.on_trail_error(hyper_model, space_sample, trail_no)

        def on_summary():
            t = hyper_model.get_best_trail()
            if t:
                detail = f'reward={t.reward}, trail_no={t.trail_no}, space_id={t.space_sample.space_id}'
                return f'best: {detail}'
            else:
                return None

        print(f'start driver server at {self.address}')
        from .grpc.SearchDriverService import serve
        server, search_service = serve(self.address, self.spaces_dir,
                                       on_next=on_next_space,
                                       on_report=on_report_space,
                                       on_summary=on_summary)

        search_start_at = time.time()

        trail_no = 1
        retry_counter = 0
        queue_size = 3
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
                            hyper_model.best_space = space_sample
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
                # print(f'push trail {trail_no} to queue')

                # wait for queued trail
                while search_service.running_size() >= queue_size:
                    time.sleep(1)
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

        print("-" * 20, 'no more space to push, start waiting...')
        while search_service.running_size() > 0:
            # print(f"wait ... {search_service.running_size()} samples found.")
            time.sleep(0.1)
        print("-" * 20, 'all trails done', '-' * 20)

        # shutdown grpc server
        server.stop(grace=1.0)
        search_service.status_thread.stop()
        search_service.status_thread.report_summary()

        # run best trail
        if hyper_model.best_space:
            space_sample = hyper_model.best_space
            print(f'run trial with best space {space_sample.space_id} in driver')
            start_at = time.time()
            trail = hyper_model._run_trial(space_sample, trail_no, X, y, X_val, y_val, **fit_kwargs)
            done_at = time.time()
            elapsed = done_at - start_at
            total_elapsed = done_at - search_start_at

            assert hyper_model.last_model
            hyper_model.best_model = hyper_model.last_model
            print(f'best model reward: {trail.reward}, elapsed={elapsed}, total elapsed={total_elapsed}')
        else:
            print(f'not found best space.')

        return trail_no
