# -*- coding:utf-8 -*-
"""

"""
from ..drift_detection import FeatureSelectorWithDriftDetection, DriftDetector


class CumlFeatureSelectorWithDriftDetection(FeatureSelectorWithDriftDetection):
    # parallelizable = False
    def _score_features(self, X_merged, y, scorer, cv):
        from . import CumlToolBox
        X_merged, y = CumlToolBox.to_local(X_merged, y)
        return super()._score_features(X_merged, y, scorer, cv)

    @staticmethod
    def get_detector(preprocessor=None, estimator=None, random_state=None):
        return CumlDriftDetector(preprocessor=preprocessor, estimator=estimator, random_state=random_state)


class CumlDriftDetector(DriftDetector):
    @staticmethod
    def _copy_data(X):
        return X.copy()
