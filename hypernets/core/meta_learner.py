# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from lightgbm import LGBMRegressor
import numpy as np


class MetaLearner(object):
    def __init__(self, history):
        self.history = history
        self.regressors = {}

    def new_sample(self, space_sample):
        self.fit(space_sample.signature)

    def fit(self, space_signature):
        features = self.history.get_features(space_signature)
        x = []
        y = []
        for f, reward in features:
            x.append(f)
            y.append(reward)
        if len(x) >= 2:
            regressor = LGBMRegressor()
            regressor.fit(x, y)
            print(regressor.predict(x))
            self.regressors[space_signature] = regressor

    def predict(self, space_sample, default_value=np.inf):
        regressor = self.regressors.get(space_sample.signature)
        if regressor is not None:
            score = regressor.predict([space_sample.features])
        else:
            score = default_value
        return score
