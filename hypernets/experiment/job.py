# -*- coding: utf-8 -*-
import abc
import os.path

import pandas as pd

from hypernets import hyperctl
from hypernets.utils import logging

logger = logging.get_logger(__name__)


class ExperimentJobCreator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_and_run_experiment(self):
        raise NotImplemented

    @staticmethod
    def _read_file(file_path):
        require_file_msg = "Csv and parquet files are supported(file name end with 'csv' or 'parquet')"
        assert file_path is not None, "file_path can not be None"
        splits = os.path.basename(file_path).split(".")
        if len(splits) < 2:
            raise ValueError(require_file_msg)
        suffix = splits[-1]
        if suffix == 'csv':
            return pd.read_csv(file_path)
        elif suffix == 'parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(require_file_msg)


class CompeteExperimentJobCreator(ExperimentJobCreator, metaclass=abc.ABCMeta):

    def create_and_run_experiment(self):
        job_params = hyperctl.get_job_params()

        if 'run_kwargs' in job_params:
            run_kwargs = job_params.pop('run_kwargs')
        else:
            run_kwargs = {}

        make_kwargs = job_params

        job_working_dir = hyperctl.api.get_job_data_dir()
        assert job_working_dir
        exp = self.create_experiment_with_params(make_kwargs, job_working_dir)
        assert exp
        exp.run(**run_kwargs)

    @staticmethod
    def set_default_eval_dir(job_working_dir, make_kwargs):
        # set default prediction dir if enable persist
        if make_kwargs.get('evaluation_persist_prediction') is True:
            if make_kwargs.get('evaluation_persist_prediction_dir') is None:
                make_kwargs['evaluation_persist_prediction_dir'] = f"{job_working_dir}/prediction"

    @staticmethod
    def set_default_render_path(job_working_dir, make_kwargs):
        # set default report file path
        if make_kwargs.get('report_render') == 'excel':
            report_render_options = make_kwargs.get('report_render_options')
            default_excel_report_path = f"{job_working_dir}/report.xlsx"
            if report_render_options is not None:
                if report_render_options.get('file_path') is None:
                    report_render_options['file_path'] = default_excel_report_path
                    make_kwargs['report_render_options'] = report_render_options
                else:
                    pass  # use user setting
            else:
                make_kwargs['report_render_options'] = {'file_path': default_excel_report_path}
        return make_kwargs

    def create_experiment_with_params(self, make_kwargs, job_working_dir):

        self.set_default_eval_dir(job_working_dir, make_kwargs)
        self.set_default_render_path(job_working_dir, make_kwargs)

        return self._create_experiment(make_kwargs)

    @abc.abstractmethod
    def _create_experiment(self, make_options):
        raise NotImplemented
