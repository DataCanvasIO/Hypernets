import os.path

import pandas as pd
import pytest
import yaml

from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment.job import run, JobEngine, DatasetConf, CompeteExperimentJobEngine
from hypernets.utils import const, common as common_util
from hypernets.tabular.datasets import dsutils


class TestJobEngine:

    def test_flat_dict(self):
        data_dict = {
            'log_level': 20,
            'evaluation': {
                'metrics': 'auto',
                'persist_prediction': True
            },
            'custom': {
                'k1': 'v1'
            }
        }
        ret_dict = CompeteExperimentJobEngine._flat_compete_experiment_conf(data_dict)
        assert ret_dict['log_level'] == data_dict['log_level']
        assert ret_dict['evaluation_metrics'] == data_dict['evaluation']['metrics']
        assert ret_dict['evaluation_persist_prediction'] == data_dict['evaluation']['persist_prediction']
        assert ret_dict['custom']['k1'] == data_dict['custom']['k1']

    def test_read_txt_file(self):
        from hypernets.tabular.datasets.dsutils import basedir
        with pytest.raises(ValueError):
            txt_file = f'{basedir}/movielens_sample.txt'
            JobEngine._read_file(txt_file)

    def test_read_supported_file(self):
        from hypernets.tabular.datasets.dsutils import basedir
        csv_file = f'{basedir}/heart-disease-uci.csv'
        df_csv = JobEngine._read_file(csv_file)
        assert df_csv.shape[0] > 1

        file_path = common_util.get_temp_file_path(prefix="heart-disease-uci", suffix=".parquet")
        df_csv.to_parquet(file_path)

        df_parquet = pd.read_parquet(file_path)
        assert df_parquet.shape == df_csv.shape


class BloodDatasetJobEngine(CompeteExperimentJobEngine):

    def _create_experiment(self, make_options):
        from hypernets.experiment import make_experiment
        dateset_conf: DatasetConf = self.job_conf.dataset

        assert dateset_conf.target == "Class"
        assert self.job_conf.run_options['max_trials'] == 2

        assert make_options['feature_selection'] is True  # asset convert report.enable to param report=True
        assert make_options['feature_selection_strategy'] == "threshold"

        assert make_options['evaluation_metrics'] == 'auto'  # asset concat keys
        assert make_options['evaluation_persist_prediction'] is True  # asset convert camel to snake

        train_data = self._read_file(dateset_conf.train_file)
        make_options['eval_data'] = self._read_file(dateset_conf.eval_file)
        make_options['test_data'] = self._read_file(dateset_conf.test_file)

        make_options['target'] = dateset_conf.target
        make_options['task'] = dateset_conf.task

        make_options['search_space'] = PlainSearchSpace()

        experiment = make_experiment(PlainModel, train_data, **make_options)

        return experiment


class TestRunJob:

    def create_config(self):
        working_dir = common_util.get_temp_dir_path()
        df = dsutils.load_blood()
        data_file = common_util.get_temp_file_path(suffix='.parquet')
        df.to_parquet(data_file)

        dict_data = {
            "version": 2.5,
            "workingDir": working_dir,
            "dataset": {
                "target": "Class",
                "task": const.TASK_BINARY,
                "trainFile": data_file,
                "evalFile": data_file,
                "testFile": data_file
            },
            "jobs": [
                {
                    "name": "blood",
                    "engine": "plain",
                    "experiment": {
                        "evaluation": {
                            "metrics": "auto",
                            "persistPrediction": True
                        },
                        'report': {
                            'render': 'excel'
                        },
                        'featureSelection': {
                            'enable': True,
                            'strategy': 'threshold'
                        }

                    },
                    "runOptions": {
                        "max_trials": 2
                    }
                }
            ],

        }
        config_p = common_util.get_temp_file_path(suffix='.yaml')
        with open(config_p, 'w') as f:
            yaml.dump(dict_data, f, yaml.CDumper)
        assert os.path.exists(config_p)
        return config_p

    def test_job(self):
        p = self.create_config()
        engines = {'plain': BloodDatasetJobEngine}
        run(p, engines)
