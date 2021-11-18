# -*- coding:utf-8 -*-
"""

"""
import os
import shutil

import pandas as pd
import numpy as np
from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import make_experiment
from hypernets.tabular.datasets import dsutils
from . import if_cuml_ready, is_cuml_installed
from ... import test_output_dir

if is_cuml_installed:
    import cudf
    from hypernets.tabular.cuml_ex import CumlToolBox


def run_experiment(train_df, **kwargs):
    experiment = make_experiment(PlainModel, train_df, search_space=PlainSearchSpace(), **kwargs)
    estimator = experiment.run()
    print(experiment.random_state, estimator)
    assert estimator is not None


@if_cuml_ready
class TestCumlExperiment:
    work_dir = f'{test_output_dir}/TestCumlExperiment'

    @classmethod
    def setup_class(cls):
        from sklearn.preprocessing import LabelEncoder
        df = dsutils.load_bank()
        df['y'] = LabelEncoder().fit_transform(df['y'])  # binary task target
        df['education'] = LabelEncoder().fit_transform(df['education'])  # multiclass task target
        cls.bank_data_cudfbank_data = df
        cls.bank_data_cudf = cudf.from_pandas(df)

        cls.boston_data = dsutils.load_blood()
        cls.boston_data_cudf = cudf.from_pandas(cls.boston_data)

        cls.movie_lens = dsutils.load_movielens()

        os.makedirs(cls.work_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_regression(self):
        run_experiment(self.boston_data_cudf.copy(),
                       cv=False,
                       ensemble_size=0,
                       max_trials=5,
                       # log_level='info',
                       random_state=335, )

    def test_multiclass(self):
        preprocessor = CumlToolBox.general_preprocessor(self.bank_data_cudf)
        run_experiment(self.bank_data_cudf.copy(), target='education',
                       hyper_model_options=dict(transformer=preprocessor),
                       cv=True,
                       ensemble_size=5,
                       max_trials=5,
                       # log_level='info',
                       random_state=335, )

    def test_binary(self):
        preprocessor = CumlToolBox.general_preprocessor(self.bank_data_cudf)
        run_experiment(self.bank_data_cudf.copy(),
                       hyper_model_options=dict(transformer=preprocessor),
                       cv=False,
                       ensemble_size=5,
                       max_trials=5,
                       # log_level='info',
                       random_state=335, )

    def test_binary_cv(self):
        preprocessor = CumlToolBox.general_preprocessor(self.bank_data_cudf)
        run_experiment(self.bank_data_cudf.copy(),
                       hyper_model_options=dict(transformer=preprocessor),
                       # cv=False,
                       ensemble_size=5,
                       max_trials=5,
                       # log_level='info',
                       random_state=335, )
