import pandas as pd
from pathlib import Path

import pytest

from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import CompeteExperiment
from hypernets.experiment.job import ExperimentJobCreator, CompeteExperimentJobCreator
from hypernets.tabular.datasets import dsutils
from hypernets.utils import const, common as common_util

try:
    from pandas.io.parquet import get_engine

    get_engine('auto')
    is_pd_parquet_ready = True
except:
    is_pd_parquet_ready = False


class BloodDatasetJobEngine(CompeteExperimentJobCreator):

    def _create_experiment(self, make_options):
        from hypernets.experiment import make_experiment
        train_data = dsutils.load_blood()
        experiment = make_experiment(PlainModel, train_data, **make_options)
        return experiment


class TestExperimentJobCreator:

    def test_read_txt_file(self):
        from hypernets.tabular.datasets.dsutils import basedir
        with pytest.raises(ValueError):
            txt_file = f'{basedir}/movielens_sample.txt'
            ExperimentJobCreator._read_file(txt_file)

    @pytest.mark.skipif(not is_pd_parquet_ready, reason='pandas parquet engine is not ready')
    def test_read_supported_file(self):
        from hypernets.tabular.datasets.dsutils import basedir
        csv_file = f'{basedir}/heart-disease-uci.csv'
        df_csv = ExperimentJobCreator._read_file(csv_file)
        assert df_csv.shape[0] > 1

        file_path = common_util.get_temp_file_path(prefix="heart-disease-uci", suffix=".parquet")
        df_csv.to_parquet(file_path)

        df_parquet = pd.read_parquet(file_path)
        assert df_parquet.shape == df_csv.shape

    def test_set_default_eval_dir(self):
        kwargs_ = {
            'evaluation_persist_prediction': True
        }
        CompeteExperimentJobCreator.set_default_eval_dir("/tmp", kwargs_)
        assert kwargs_['evaluation_persist_prediction_dir'] == "/tmp/prediction"

    def test_set_default_render_path(self):
        kwargs_ = {
            'report_render': 'excel'
        }
        kwargs__ = CompeteExperimentJobCreator.set_default_render_path("/tmp", kwargs_)
        assert kwargs__['report_render_options'].get('file_path') == "/tmp/report.xlsx"

    def test_creator(self):
        test_data = dsutils.load_blood()
        eval_data = dsutils.load_blood()
        make_options = {
            'test_data': test_data,
            'eval_data': eval_data,
            "task": const.TASK_BINARY,
            'target': "Class",
            'feature_selection': True,
            'feature_selection_strategy': "threshold",
            "evaluation_metrics": "auto",
            "evaluation_persist_prediction": True,
            "report_render": 'excel',
            "search_space": PlainSearchSpace(),
        }
        job_working_dir = common_util.get_temp_dir_path(prefix="hyn_job_creator_test_")

        exp = BloodDatasetJobEngine().create_experiment_with_params(make_options, job_working_dir)
        assert exp
        assert isinstance(exp, CompeteExperiment)

        run_options = {
            "max_trials": 2
        }

        exp.run(**run_options)
        assert (Path(job_working_dir) / "report.xlsx").exists()
