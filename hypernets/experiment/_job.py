# -*- coding: utf-8 -*-
import abc
import os.path
import pkg_resources
import argparse
from typing import Type, List
import json

import pandas as pd
import yaml

from hypernets.utils import common as common_util
from hypernets.utils import logging

logger = logging.get_logger(__name__)


class DatasetConf:
    def __init__(self, train_file, target, task=None, eval_file=None, test_file=None):
        self.train_file = train_file
        self.target = target
        self.task = task
        self.eval_file = eval_file
        self.test_file = test_file


class JobConf:
    def __init__(self, name, engine, dataset, working_dir, experiment=None, run_options=None, webui=None):
        assert name is not None, "name can not be None"
        assert dataset is not None, "dataset can not be None"
        self.name = name
        self.engine = engine
        self.dataset = dataset
        self.working_dir = working_dir
        self.experiment = experiment
        self.run_options = run_options
        self.webui = webui


class JobEngine(metaclass=abc.ABCMeta):

    def __init__(self, job_conf: JobConf):
        self.job_conf = job_conf

    @abc.abstractmethod
    def create_experiment(self):
        raise NotImplemented

    @staticmethod
    @abc.abstractmethod
    def name():
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
    STATUS_INIT = 'INIT'
    STATUS_RUNNING = 'RUNNING'
    STATUS_SUCCEED = 'SUCCEED'
    STATUS_FAILED = 'FAILED'

    def __init__(self, conf: JobConf):
        self.conf = conf
        self.status_ = Job.STATUS_INIT


class JobGroup:
    def __init__(self, dataset_conf: DatasetConf, jobs_conf: List[JobConf], working_dir: str):
        self.dataset_conf = dataset_conf
        # validate name is unique in jobs
        counter = {}
        for job_conf in jobs_conf:
            counter[job_conf.name] = counter.get(job_conf.name, 0) + 1
        for k, v in counter.items():
            duplicate = []
            if v > 1:
                duplicate.append(k)
            if len(duplicate) > 0:
                raise ValueError(f"Jobs name is duplicate: { ','.join(duplicate)} ")
        self.jobs_conf = jobs_conf
        self.working_dir = working_dir


class ConfigLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def _load_as_yaml(self):
        with open(self.file_path, 'r') as f:
            content = f.read()
        return yaml.load(content, Loader=yaml.CLoader)

    def load(self) -> JobGroup:
        config_dict = self._load_as_yaml()
        config_dict_snake_case = common_util.camel_keys_to_snake(config_dict)

        dataset_conf = DatasetConf(**config_dict_snake_case['dataset'])
        job_dict_list = config_dict_snake_case['jobs']

        working_dir = config_dict_snake_case.get('working_dir', '.')
        working_dir = os.path.abspath(working_dir)
        logger.info(f"Working dir at: {working_dir}")
        if not os.path.exists(working_dir):
            logger.info("Create working dir")
            os.makedirs(working_dir, exist_ok=True)

        job_conf_list = []
        for job_dict in job_dict_list:
            job_init_kwargs = {'dataset': dataset_conf, 'working_dir': working_dir, **job_dict}
            job_conf_list.append(JobConf(**job_init_kwargs))

        return JobGroup(dataset_conf=dataset_conf, jobs_conf=job_conf_list, working_dir=working_dir)


class JobGroupControl:
    def __init__(self, job_group: JobGroup, engines_cls: List[Type[JobEngine]]):
        self.job_group = job_group
        self.engines_cls_dict = {engine_cls.name(): engine_cls for engine_cls in engines_cls}

        self.jobs_: List[Job] = [Job(_) for _ in self.job_group.jobs_conf]

    def run(self):
        for i, job in enumerate(self.jobs_):
            try:
                logger.info(f"Start to run job {i+1}/{len(self.job_group.jobs_conf)}.")
                job_conf: JobConf = job.conf
                engine_cls = self.engines_cls_dict.get(job_conf.engine)
                if engine_cls is None:
                    logger.error(f"Engine '{job_conf.engine}' specified "
                                 f"in job '{job_conf.name}' not found, skipped this job.")
                    continue
                exp = engine_cls(job_conf).create_experiment()
                exp.run(**job_conf.run_options)
                job.status_ = Job.STATUS_SUCCEED
            except Exception as e:
                job.status_ = Job.STATUS_FAILED
                logger.exception(e)
        summary = {job.conf.name: job.status_ for job in self.jobs_}
        logger.info("Jobs summary:\n" + json.dumps(summary))


class JobGroupControlCLI:

    def run(self, config_path):
        dev_mode = True
        entry_point_name = 'hypernets_experiment'

        # load plugin, https://setuptools.pypa.io/en/latest/pkg_resources.html#entry-points
        plugins = []
        if not dev_mode:
            for entrypoint in pkg_resources.iter_entry_points(group=entry_point_name):
                plugin = entrypoint.load()
                # TODO: info plugin
                plugins[plugin.get_module_name()] = plugin
        else:
            from hypergbm.job import HyperGBMJobEngine
            plugins = [HyperGBMJobEngine]

        # read config
        job_group = ConfigLoader(config_path).load()

        # run the jobs
        jgc = JobGroupControl(job_group, plugins)
        jgc.run()

    def main(self):
        parser = argparse.ArgumentParser(description='hyperctl command is used to manage experiments', add_help=True)
        subparsers = parser.add_subparsers(dest="operation")

        exec_parser = subparsers.add_parser("run", help="run experiments job")
        exec_parser.add_argument("-c", "--config", help="yaml config file path", default=None, required=True)
        args_namespace = parser.parse_args()
        print(args_namespace)
        operation = args_namespace.operation
        if operation == 'run':
            config_file = args_namespace.config
            self.run(config_path=config_file)
        else:
            parser.print_help()
