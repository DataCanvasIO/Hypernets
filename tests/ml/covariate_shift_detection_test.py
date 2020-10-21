# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.frameworks.ml.datasets.dsutils import load_bank
from hypernets.frameworks.ml.utils.shift_detection import covariate_shift_score


class Test_covariate_shift_detection:
    def test_shift_score(self):
        df = load_bank().head(1000)
        scores = covariate_shift_score(df[:700], df[700:])
        assert scores['id'] > 0.8
