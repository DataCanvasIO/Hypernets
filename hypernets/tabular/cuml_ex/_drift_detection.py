# -*- coding:utf-8 -*-
"""

"""
from ..drift_detection import FeatureSelectorWithDriftDetection, DriftDetector


class CumlFeatureSelectorWithDriftDetection(FeatureSelectorWithDriftDetection):
    parallelizable = False

    @staticmethod
    def get_detector(preprocessor=None, estimator=None, random_state=None):
        return CumlDriftDetector(preprocessor=preprocessor, estimator=estimator, random_state=random_state)


class CumlDriftDetector(DriftDetector):
    @staticmethod
    def _copy_data(X):
        return X.copy()
