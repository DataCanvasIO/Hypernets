# -*- coding:utf-8 -*-
"""

"""

from hypernets.utils import logging
from ..drift_detection import FeatureSelectorWithDriftDetection, DriftDetector

logger = logging.getLogger(__name__)


class DaskFeatureSelectionWithDriftDetector(FeatureSelectorWithDriftDetection):
    parallelizable = False

    @staticmethod
    def get_detector(preprocessor=None, estimator=None, random_state=9527):
        return DaskDriftDetector(preprocessor=preprocessor, estimator=estimator, random_state=random_state)


class DaskDriftDetector(DriftDetector):
    @staticmethod
    def _copy_data(X):
        return X.copy()
