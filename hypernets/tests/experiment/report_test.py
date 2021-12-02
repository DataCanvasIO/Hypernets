import os
import datetime
import pandas as pd
import pytest

from hypernets.experiment.report import ExperimentReport, ExcelReportRender


class Test_Excel_Report():

    @staticmethod
    def create_prediction_stats_df():
        return pd.DataFrame(data={
            'dataset': ['Test'],
            'elapsed': [1000],
            'rows': [10000000],
            'speed': [100000/1024]  # {n}K/s
        })

    @staticmethod
    def create_feature_trans_df():
        return pd.DataFrame(data={
            'name': ["Age", "Sex", "Name", "Geight", "Other"],
            'col': ["age", "sex", "name", "height", "Other"],
            'method': ["drop", "drop", "drop", "drop", "drop"],
            'stage': ["DataCleanStep", "DriftDetectStep", "FeatureImportanceSelectionStep",
                      "FeatureImportanceSelectionStep", "Unknown"],
            'reason': ["constant", "constant", "constant", "constant", "constant"],
            'remark': ["Empty", "Empty", "Empty", "Empty", "Empty"]
        })

    @staticmethod
    def create_dataset_df():
        return pd.DataFrame(data={
            'type': ["Train"],
            'task': ["binary-classification"],
            'shape': [['100', '200']],
            'memory': [1024],
            'size': [1024],
            'has_label': [True],
            'file_type': ["csv"],
            'remark': ["Empty"]
        })

    @staticmethod
    def create_confusion_matrix_df():
        df = pd.DataFrame(data={
            'A': [1, 4, 6],
            'B': [1, 4, 6],
            'C': [1, 4, 6],
        })
        # df.set_index(df.columns)
        df.index = df.columns
        return df

    @staticmethod
    def create_resource_monitor_df():
        now = datetime.datetime.now()
        s_10 = datetime.timedelta(seconds=10)
        df = pd.DataFrame(data={
            'datetime': [now, now + s_10, now + s_10 + s_10],
            'cpu': [100, 200, 300],
            'ram': [3000, 3500, 4000]
        })
        return df

    @staticmethod
    def create_feature_importances_df():
        imps = pd.DataFrame(data={
            'col_name': ['age', 'name'],
            'feature': ['Age', 'Name'],
            'importances': [0.3, 0.2]
        })
        m1 = {
            'weight': 0.2,
            'lift': 0.2,
            'models': [imps.copy(), imps.copy(), imps.copy()]
        }
        m2 = {
            'weight': 0.3,
            'lift': 0.1,
            'models': [imps.copy(), imps.copy(), imps.copy()]
        }
        return [m1, m2]

    @staticmethod
    def create_binary_metric_df():
        return pd.DataFrame(data={
            '': [0, 1, 2, 'macro avg', 'weighted avg'],
            'precision': [0.8, 0.9, 0.9, 0.095, 0.098],
            'recall': [0.8, 0.9, 0.9, 0.095, 0.098],
            'f1-score': [0.8, 0.9, 0.9, 0.095, 0.098],
            'support': [100, 200, 100, 300, 400]
        })

    @staticmethod
    def _get_file_path():
        def cast_int(v):
            try:
                return int(v.split(".")[0])
            except Exception as e:
                return 0
        max_num = max([cast_int(name) for name in os.listdir("Z:/excel")])
        file_path = f"Z:/excel/{max_num+1}.xlsx"
        return file_path

    def test_render(self):
        er = ExperimentReport(datasets=self.create_dataset_df(),
                              feature_trans=self.create_feature_trans_df(),
                              evaluation_metric=self.create_binary_metric_df(),
                              confusion_matrix=self.create_confusion_matrix_df(),
                              resource_usage=self.create_resource_monitor_df(),
                              feature_importances=self.create_feature_importances_df(),
                              prediction_stats=self.create_prediction_stats_df())
        p = self._get_file_path()
        print('excel write to: ' + p)
        ExcelReportRender(p).render(er)
        assert os.path.exists(p)

    # @pytest.mark.xfail(raises=AssertionError)
    def test_df_missing_key(self):
        with pytest.raises(AssertionError) as excinfo:
            df = self.create_dataset_df()
            df.drop('type', axis=1, inplace=True)
            er = ExperimentReport(datasets=df)
            p = self._get_file_path()
            print('excel write to: ' + p)
            ExcelReportRender(p).render(er)

    def test_imps_missing_key(self):
        with pytest.raises(AssertionError) as excinfo:
            estimators = self.create_feature_importances_df()
            del estimators[0]['models'][0]['col_name']
            er = ExperimentReport(feature_importances=estimators)
            p = self._get_file_path()
            print('excel write to: ' + p)
            ExcelReportRender(p).render(er)
