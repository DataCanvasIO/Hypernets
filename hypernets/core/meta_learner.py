# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from lightgbm import LGBMRegressor
import numpy as np
from ..utils import logging

logger = logging.get_logger(__name__)


class MetaLearner(object):
    def __init__(self, history, dataset_id, trial_store):
        self.trial_store = trial_store
        self.dataset_id = dataset_id
        self.history = history
        self.regressors = {}
        self.store_history = {}

        if logger.is_info_enabled():
            logger.info(f'Initialize Meta Learner: dataset_id:{dataset_id}')

    def new_sample(self, space_sample):
        self.fit(space_sample.signature)

    def fit(self, space_signature):

        features = self.extract_features_and_labels(space_signature)
        x = []
        y = []
        for features, label in features:
            x.append(features)
            y.append(label)

        store_history = self.store_history.get(space_signature)

        if self.trial_store is not None and store_history is None:
            trials = self.trial_store.get_all(self.dataset_id, space_signature)
            store_x = []
            store_y = []
            for t in trials:
                store_x.append(t.space_sample_vectors)
                store_y.append(t.reward)
            store_history = (store_x, store_y)
            self.store_history[space_signature] = store_history

        if store_history is None:
            store_history = ([], [])

        store_x, store_y = store_history
        x = x + store_x
        y = y + store_y
        if len(x) >= 2:
            regressor = LGBMRegressor()
            regressor.fit(x, y)
            #  if logger.is_info_enabled():
            #      logger.info(regressor.predict(x))
            self.regressors[space_signature] = regressor

    def predict(self, space_sample, default_value=np.inf):
        regressor = self.regressors.get(space_sample.signature)
        if regressor is not None:
            score = regressor.predict([space_sample.vectors])
        else:
            score = default_value
        return score

    def extract_features_and_labels(self, signature):
        features = [(t.space_sample.vectors, t.reward) for t in self.history.history if
                    t.space_sample.signature == signature]
        return features
