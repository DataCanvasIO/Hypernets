# -*- coding: utf-8 -*-
import abc
import os.path
import pandas as pd
import yaml
import pkg_resources
from typing import Type
from hypernets.utils import common as common_util

from hypernets.utils import logging

logger = logging.get_logger(__name__)


class DatasetConf:
    def __init__(self, task, target, train_file, eval_file=None, test_file=None):
        self.task = task
        self.target = target
        self.train_file = train_file
        self.eval_file = eval_file
        self.test_file = test_file


class WebUIConf:
    def __init__(self, ip="0.0.0.0", port=8866):
        self.ip = ip
        self.port = port

    @staticmethod
    def load_from_dict(d):
        return WebUIConf(**d)


class JobConf:
    def __init__(self, name, dataset, experiment=None, run_options=None, webui=None):
        self.name = name
        self.dataset = dataset
        self.experiment = experiment
        self.run_options = run_options
        self.webui = webui


class JobApp(metaclass=abc.ABCMeta):

    def __init__(self, job_conf: JobConf):
        self.job_conf = job_conf

    @abc.abstractmethod
    def create_experiment(self):
        raise NotImplemented

    @staticmethod
    @abc.abstractmethod
    def module_name():
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

    @staticmethod
    def _flat_compete_experiment_conf(dict_data):
        """
        feature_selection=False,
                 feature_selection_strategy=None,
        example:
            Convert
                {
                    'feature_selection': {
                        'enable': True,
                        'strategy': 'threshold'
                    }
                }
            to
                {
                    'feature_selection': True,
                    'feature_selection_strategy': 'threshold'
                }
        Parameters
        ----------
        dict_data

        Returns
        -------

        """
        enable_key = 'enable'

        sub_config_keys = ['feature_generation', 'drift_detection', 'feature_selection', 'down_sample_search',
                           'feature_reselection', 'pseudo_labeling', 'report', 'early_stopping', 'evaluation']
        ret_dict = {}
        for config_key, config_value in dict_data.items():
            if config_key in sub_config_keys:
                flat_dict = {}
                if enable_key in config_value:
                    flat_dict[config_key] = config_value[enable_key]
                for k, v in config_value.items():
                    if k != enable_key:
                        flat_dict[f'{config_key}_{k}'] = v
                ret_dict.update(flat_dict)
            else:
                ret_dict[config_key] = config_value

        return ret_dict


class Job:
    def __init__(self, conf):
        self.conf = conf

        self._status = None


class JobGroup:
    def __init__(self, dataset_conf: DatasetConf, jobs_conf):
        self.dataset_conf = dataset_conf
        self.jobs_conf = jobs_conf


class ConfigLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def _load_as_yaml(self):
        with open(self.file_path, 'r') as f:
            content = f.read()
        return yaml.load(content, Loader=yaml.CLoader)

    def load(self, ) -> JobGroup:
        config_dict = self._load_as_yaml()
        config_dict_snake_case = common_util.camel_keys_to_snake(config_dict)

        dataset_conf = DatasetConf(**config_dict_snake_case['dataset'])
        job_dict_list = config_dict_snake_case['jobs']

        job_conf_list = []
        for job_dict in job_dict_list:
            job_init_kwargs = {'dataset': dataset_conf, **job_dict}
            job_conf_list.append(JobConf(**job_init_kwargs))

        return JobGroup(dataset_conf=dataset_conf,
                        jobs_conf=job_conf_list)


class JobGroupControl:
    def __init__(self, job_group: JobGroup, job_plugin_cls: Type[JobApp]):
        self.job_group = job_group
        self.job_plugin_cls = job_plugin_cls

    def run(self):
        for i, job_conf in enumerate(self.job_group.jobs_conf):
            try:
                logger.info(f"Start to run job {i+1}/{len(self.job_group.jobs_conf)}.")
                job_conf: JobConf = job_conf
                job = Job(job_conf)
                exp = self.job_plugin_cls(job_conf).create_experiment()
                exp.run(**job_conf.run_options)
            except Exception as e:
                logger.exception(e)

    def list(self):
        pass

    def stop(self, ):
        pass


class JobGroupControlCLI:

    def main(self, sub_module, config_path):
        dev_mode = True
        entry_point_name = 'hypernets_experiment'

        # load plugin, https://setuptools.pypa.io/en/latest/pkg_resources.html#entry-points
        plugins = []
        if not dev_mode:
            for entrypoint in pkg_resources.iter_entry_points(group=entry_point_name):
                plugin = entrypoint.load()
                plugins[plugin.get_module_name()] = plugin
        else:
            from hypergbm.job import HyperGBMJobApp
            plugins = [HyperGBMJobApp]

        plugins_dict = {p.module_name(): p for p in plugins}
        assert sub_module in plugins_dict, f"App '{sub_module}' not exits "

        plugin_cls = plugins_dict[sub_module]

        # read config
        job_group = ConfigLoader(config_path).load()

        # run the jobs
        jgc = JobGroupControl(job_group, plugin_cls)
        jgc.run()
