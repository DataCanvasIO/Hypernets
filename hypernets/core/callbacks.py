# -*- coding:utf-8 -*-
"""

"""
import time


class Callback():
    def __init__(self):
        pass

    def on_build_estimator(self, hyper_model, space, estimator, trail_no):
        pass

    def on_trail_begin(self, hyper_model, space, trail_no):
        pass

    def on_trail_end(self, hyper_model, space, trail_no, reward, improved, espl):
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


class SummaryCallback(Callback):
    def on_build_estimator(self, hyper_model, space, estimator, trail_no):
        print(f'\nTrail No:{trail_no}')
        space.params_summary()
        estimator.summary()

    def on_trail_begin(self, hyper_model, space, trail_no):
        print('trail begin')

    def on_trail_end(self, hyper_model, space, trail_no, reward, improved, elapsed):
        print(f'trail end. reward:{reward}, improved:{improved}, elapsed:{elapsed}')
        print(f'Total elapsed:{time.time() - hyper_model.start_search_time}')
