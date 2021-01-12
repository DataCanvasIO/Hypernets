# -*- coding:utf-8 -*-
"""

"""


class Dispatcher(object):
    def __init__(self):
        super(Dispatcher, self).__init__()

    def dispatch(self, hyper_model, X, y, X_val, y_val, cv, num_folds, max_trials, dataset_id, trial_store,
                 **fit_kwargs):
        raise NotImplemented()

    # def run_trial(self, space_sample, trial_no, X, y, X_val, y_val, **fit_kwargs):
    #     pass
