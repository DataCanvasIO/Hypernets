# -*- coding:utf-8 -*-
"""

"""


class Dispatcher(object):
    def __init__(self):
        super(Dispatcher, self).__init__()

    def dispatch(self, hyper_model, X, y, X_val, y_val, max_trails, dataset_id, trail_store,
                 **fit_kwargs):
        raise NotImplemented()

    # def run_trail(self, space_sample, trail_no, X, y, X_val, y_val, **fit_kwargs):
    #     pass
