import datetime
import os
import tempfile
import time

import pandas as pd

from hypernets.experiment.compete import StepNames
from hypernets.experiment.report import ExcelReportRender
from hypernets.utils import const
from hypernets.experiment import ExperimentMeta, DatasetMeta, StepMeta, StepType
from hypernets.experiment import ResourceUsageMonitor


class TestExcelReport:

    @staticmethod
    def create_prediction_stats_df():
        return pd.DataFrame(data={
            'dataset': ['Test'],
            'elapsed': [1000],
            'rows': [10000000],
            'speed': [100000/1024]  # {n}K/s
        })

    @staticmethod
    def create_dataset_meta():
        dm1 = DatasetMeta(kind="Train", task=const.TASK_BINARY, shape=[100, 200], memory=1024)
        return [dm1]

    @staticmethod
    def create_confusion_matrix_data():
        return [[1, 4, 6], [1, 4, 6], [1, 4, 6]]

    @staticmethod
    def create_resource_monitor_df():
        now = datetime.datetime.now()
        s_10 = datetime.timedelta(seconds=10)
        return [
            (now, 100, 3000),
            (now + s_10, 100, 3500),
            (now + s_10 + s_10, 300, 4000),
        ]

    @staticmethod
    def create_ensemble_step_meta():
        imps = {'Age': 0.3, 'Name': 0.2}
        m1 = {
            'index': 0,
            'weight': 0.2,
            'lift': 0.2,
            'models': [imps.copy(), imps.copy(), imps.copy()]
        }
        m2 = {
            'index': 1,
            'weight': 0.3,
            'lift': 0.1,
            'models': [imps.copy(), imps.copy(), imps.copy()]
        }
        extension = {
            'estimators': [m1, m2]
        }
        s_meta = StepMeta(index=0,
                          name=StepNames.ENSEMBLE,
                          type=StepType.Ensemble,
                          status=StepMeta.STATUS_FINISH,
                          configuration={},
                          extension=extension,
                          start_datetime=datetime.datetime.now(),
                          end_datetime=datetime.datetime.now())
        return s_meta

    @staticmethod
    def create_binary_metric_data():
        return {
            '0': {'precision': 0.8, 'recall': 0.8, 'f1-score': 0.8, 'support': 0.8},
            '1': {'precision': 0.8, 'recall': 0.8, 'f1-score': 0.8, 'support': 0.8},
            '2': {'precision': 0.8, 'recall': 0.8, 'f1-score': 0.8, 'support': 0.8},
            'macro avg': {'precision': 0.8, 'recall': 0.8, 'f1-score': 0.8, 'support': 0.8},
            'weighted avg': {'precision': 0.8, 'recall': 0.8, 'f1-score': 0.8, 'support': 0.8},
        }

    @staticmethod
    def create_data_clean_step_meta():
        extension = {
            'unselected_reason': {
                'Name': 'constant',
                'Id': 'idness',
            }
        }
        s_meta = StepMeta(index=0,
                          name=StepNames.DATA_CLEAN,
                          type=StepType.DataCleaning,
                          status=StepMeta.STATUS_FINISH,
                          configuration={},
                          extension=extension,
                          start_datetime=datetime.datetime.now(),
                          end_datetime=datetime.datetime.now())
        return s_meta

    @staticmethod
    def _get_file_path():
        file_path = tempfile.mkstemp(prefix="report_excel_", suffix=".xlsx")[1]
        return file_path

    def test_render(self):
        steps_meta = [self.create_data_clean_step_meta(), self.create_ensemble_step_meta()]
        experiment_meta = ExperimentMeta(task=const.TASK_BINARY,
                                         datasets=self.create_dataset_meta(),
                                         steps=steps_meta,
                                         evaluation_metric=self.create_binary_metric_data(),
                                         confusion_matrix=self.create_confusion_matrix_data(),
                                         resource_usage=self.create_resource_monitor_df(),
                                         prediction_stats=self.create_prediction_stats_df())
        p = self._get_file_path()
        print(p)
        ExcelReportRender(file_path=p).render(experiment_meta)
        assert os.path.exists(p)


class TestResourceUsageMonitor:

    def test_sample(self):
        rum = ResourceUsageMonitor(1)
        rum.start_watch()
        time.sleep(2)
        rum.stop_watch()
        assert len(rum.data) > 1
        assert rum._timer_status == ResourceUsageMonitor.STATUS_STOP


# class TestFeatureTransCollector:
#
#     def test_collector(self):
#         pass
