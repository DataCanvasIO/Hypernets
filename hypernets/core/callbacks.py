# -*- coding:utf-8 -*-
"""

"""
import time
import os
import datetime
import json


class Callback():
    def __init__(self):
        pass

    def on_build_estimator(self, hyper_model, space, estimator, trail_no):
        pass

    def on_trail_begin(self, hyper_model, space, trail_no):
        pass

    def on_trail_end(self, hyper_model, space, trail_no, reward, improved, elapsed):
        pass

    def on_skip_trail(self, hyper_model, space, trail_no, reason, reward, improved, elapsed):
        pass


class EarlyStoppingError(RuntimeError):
    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass


class EarlyStopping(Callback):
    def __init__(self):
        pass


class FileLoggingCallback(Callback):
    def __init__(self, searcher, output_dir=None):
        self.output_dir = self._prepare_output_dir(output_dir, searcher)

    def _prepare_output_dir(self, log_dir, searcher):
        if log_dir is None:
            log_dir = 'log'
        if log_dir[-1] == '/':
            log_dir = log_dir[:-1]

        running_dir = f'exp_{searcher.__class__.__name__}_{datetime.datetime.now().__format__("%m%d-%H%M%S")}'
        output_path = os.path.expanduser(f'{log_dir}/{running_dir}/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path

    def on_build_estimator(self, hyper_model, space, estimator, trail_no):
        pass

    def on_trail_begin(self, hyper_model, space, trail_no):
        pass
        # with open(f'{self.output_dir}/trail_{trail_no}.log', 'w') as f:
        #     f.write(space.params_summary())

    def on_trail_end(self, hyper_model, space, trail_no, reward, improved, elapsed):
        with open(f'{self.output_dir}/trail_{improved}_{trail_no:04d}_{reward:010.8f}_{elapsed:06.2f}.log', 'w') as f:
            f.write(space.params_summary())
            f.write('\r\n----------------Summary for Searcher----------------\r\n')
            f.write(hyper_model.searcher.summary())

        topn = 10
        diff = hyper_model.history.diff(hyper_model.history.get_top(topn))
        with open(f'{self.output_dir}/top_{topn}_diff.txt', 'w') as f:
            diff_str = json.dumps(diff, indent=5)
            f.write(diff_str)
            f.write('\r\n')
            f.write(hyper_model.searcher.summary())
        with open(f'{self.output_dir}/top_{topn}_config.txt', 'w') as f:
            trials = hyper_model.history.get_top(topn)
            configs = hyper_model.export_configuration(trials)
            for trail, conf in zip(trials, configs):
                f.write(f'Trail No: {trail.trail_no}, Reward: {trail.reward}\r\n')
                f.write(conf)
                f.write('\r\n---------------------------------------------------\r\n\r\n')

    def on_skip_trail(self, hyper_model, space, trail_no, reason, reward, improved, elapsed):
        with open(f'{self.output_dir}/trail_{reason}_{improved}_{trail_no:04d}_{reward:010.8f}_{elapsed:06.2f}.log',
                  'w') as f:
            f.write(space.params_summary())

        topn = 5
        diff = hyper_model.history.diff(hyper_model.history.get_top(topn))
        with open(f'{self.output_dir}/top_{topn}_diff.txt', 'w') as f:
            diff_str = json.dumps(diff, indent=5)
            f.write(diff_str)


class SummaryCallback(Callback):
    def on_build_estimator(self, hyper_model, space, estimator, trail_no):
        print(f'\nTrail No:{trail_no}')
        print(space.params_summary())
        estimator.summary()

    def on_trail_begin(self, hyper_model, space, trail_no):
        print('trail begin')

    def on_trail_end(self, hyper_model, space, trail_no, reward, improved, elapsed):
        print(f'trail end. reward:{reward}, improved:{improved}, elapsed:{elapsed}')
        print(f'Total elapsed:{time.time() - hyper_model.start_search_time}')

    def on_skip_trail(self, hyper_model, space, trail_no, reason, reward, improved, elapsed):
        print(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print(f'trail skip. reason:{reason},  reward:{reward}, improved:{improved}, elapsed:{elapsed}')
        print(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
