import datetime
import os
import time

from hypernets.experiment import ConfusionMatrixMeta
from hypernets.experiment.compete import StepNames
from hypernets.experiment.report import ExcelReportRender
from hypernets.utils import const, common as common_util
from hypernets.experiment import ExperimentMeta, DatasetMeta, StepMeta, StepType
from hypernets.experiment import ResourceUsageMonitor


class TestExcelReport:

    @staticmethod
    def create_prediction_stats_df():
        return (1000,2000)

    @staticmethod
    def create_dataset_meta():
        dm1 = DatasetMeta(kind="Train", task=const.TASK_BINARY, shape=[100, 200], memory=1024)
        return [dm1]

    @staticmethod
    def create_confusion_matrix_data():
        data = [[1, 4, 6], [1, 4, 6], [1, 4, 6]]
        labels = [0, 1, 2]
        return ConfusionMatrixMeta(data, labels)

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
    def create_ensemble_step_meta(cv=True):
        imps = {'Age': 0.3, 'Name': 0.2}

        if cv:
            models = [imps.copy(), imps.copy(), imps.copy()]
        else:
            models = [imps.copy()]

        m1 = {
            'index': 0,
            'weight': 0.2,
            'lift': 0.2,
            'models': models
        }
        m2 = {
            'index': 1,
            'weight': 0.3,
            'lift': 0.1,
            'models': models
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
    def create_classification_report():
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
    def create_pseudo_step_meta():
        extension = {
            "probabilityDensity": {},
            "samples": {"yes": 100, "no": 2000},
            "selectedLabel": "yes"
        }
        s_meta = StepMeta(index=0,
                          name=StepNames.PSEUDO_LABELING,
                          type=StepType.PseudoLabeling,
                          status=StepMeta.STATUS_FINISH,
                          configuration={},
                          extension=extension,
                          start_datetime=datetime.datetime.now(),
                          end_datetime=datetime.datetime.now())
        return s_meta

    def run_render(self, cv):
        steps_meta = [self.create_data_clean_step_meta(),
                      self.create_pseudo_step_meta(), self.create_ensemble_step_meta()]

        experiment_meta = ExperimentMeta(task=const.TASK_BINARY,
                                         datasets=self.create_dataset_meta(),
                                         steps=steps_meta,
                                         classification_report=self.create_classification_report(),
                                         confusion_matrix=self.create_confusion_matrix_data(),
                                         resource_usage=self.create_resource_monitor_df(),
                                         prediction_elapsed=self.create_prediction_stats_df())

        p = common_util.get_temp_file_path(prefix="report_excel_", suffix=".xlsx")
        print(p)
        ExcelReportRender(file_path=p).render(experiment_meta)
        assert os.path.exists(p)

    def test_enable_cv(self):
        self.run_render(True)

    def test_disable_cv(self):
        self.run_render(False)




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
