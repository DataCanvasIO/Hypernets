import os.path

import numpy as np
from sklearn.model_selection import train_test_split

from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import make_experiment, MLEvaluateCallback, MLReportCallback, ExperimentMeta
from hypernets.experiment.compete import StepNames
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from hypernets.utils import common as common_util


def test_experiment_with_blood_simple():
    df = dsutils.load_blood()
    experiment = make_experiment(PlainModel, df, target='Class', search_space=PlainSearchSpace())
    estimator = experiment.run(max_trials=3)
    print(estimator)
    assert estimator is not None


def test_experiment_with_blood_down_sample():
    df = dsutils.load_blood()
    experiment = make_experiment(PlainModel, df, target='Class', search_space=PlainSearchSpace(),
                                 down_sample_search=True, down_sample_search_size=0.1,
                                 down_sample_search_time_limit=300, down_sample_search_max_trials=10,
                                 # log_level='info',
                                 )
    estimator = experiment.run(max_trials=3)
    print(estimator)
    assert estimator is not None


def test_experiment_with_data_adaption():
    df = dsutils.load_bank()
    df = MultiLabelEncoder().fit_transform(df)
    mem_usage = int(df.memory_usage().sum())
    experiment = make_experiment(PlainModel, df, target='y', search_space=PlainSearchSpace(),
                                 data_adaption_memory_limit=mem_usage // 2,
                                 log_level='info',
                                 )
    estimator = experiment.run(max_trials=3)
    assert estimator is not None
    assert estimator.steps[0][0] == 'data_adaption'


def test_experiment_with_blood_down_sample_by_class():
    df = dsutils.load_blood()
    experiment = make_experiment(PlainModel, df, target='Class', search_space=PlainSearchSpace(),
                                 down_sample_search=True, down_sample_search_size={0: 0.2, 1: 0.5},
                                 down_sample_search_time_limit=300, down_sample_search_max_trials=10,
                                 # log_level='info',
                                 )
    estimator = experiment.run(max_trials=3)
    print(estimator)
    assert estimator is not None


def test_experiment_with_blood_full_features():
    df = dsutils.load_blood()
    target = 'Class'
    df_train, df_test = train_test_split(df, train_size=0.8, random_state=335)
    df_test.pop(target)

    experiment = make_experiment(PlainModel, df, target=target, search_space=PlainSearchSpace(),
                                 test_data=df_test,
                                 feature_generation=True,
                                 collinearity_detection=True,
                                 drift_detection=True,
                                 feature_selection=True,
                                 down_sample_search=True, down_sample_search_size=0.2,
                                 feature_reselection=True,
                                 pseudo_labeling=True,
                                 random_state=335,
                                 early_stopping_time_limit=1200,
                                 # log_level='info',
                                 )
    estimator = experiment.run(max_trials=3)
    print(estimator)
    assert estimator is not None

    step_names = [step[0] for step in estimator.steps]
    assert step_names == [StepNames.DATA_CLEAN, StepNames.MULITICOLLINEARITY_DETECTION, 'estimator']


def run_export_excel_report(maker, has_eval_data=True, str_label=True):
    df = dsutils.load_blood()
    df['Constant'] = [0 for i in range(df.shape[0])]
    df['Id'] = [i for i in range(df.shape[0])]

    target = 'Class'
    labels = ["no", "yes"]
    if str_label:
        df[target] = df[target].map(lambda v: labels[v])

    df_train, df_eval = train_test_split(df, test_size=0.2)

    df_train['Drifted'] = np.random.random(df_train.shape[0])
    df_eval['Drifted'] = np.random.random(df_eval.shape[0]) * 100

    file_path = common_util.get_temp_file_path(prefix="report_excel_", suffix=".xlsx")
    print(file_path)
    experiment = maker(df_train, target, df_eval, file_path)
    estimator = experiment.run(max_trials=3)
    assert estimator is not None
    mlr_callback = None
    mle_callback = None
    for callback in experiment.callbacks:
        if isinstance(callback, MLReportCallback):
            mlr_callback = callback
        if isinstance(callback, MLEvaluateCallback):
            mle_callback = callback

    assert mlr_callback is not None
    _experiment_meta: ExperimentMeta = mlr_callback.experiment_meta_

    assert len(_experiment_meta.resource_usage) > 0
    assert os.path.exists(file_path)

    if has_eval_data:
        assert mle_callback is not None
        assert _experiment_meta.confusion_matrix is not None
        assert _experiment_meta.classification_report is not None
        assert len(_experiment_meta.prediction_elapsed) == 2
        assert _experiment_meta.confusion_matrix.data.shape == (2, 2)  # binary classification
        assert len(_experiment_meta.datasets) == 3
    else:
        assert len(_experiment_meta.datasets) == 2
    return _experiment_meta


def test_str_render():
    def maker(df_train, target, df_eval, file_path):
        experiment = make_experiment(PlainModel, df_train,
                                     target=target,
                                     eval_data=df_eval,
                                     test_data=df_eval.copy(),
                                     drift_detection_threshold=0.4,
                                     drift_detection_min_features=3,
                                     drift_detection_remove_size=0.5,
                                     search_space=PlainSearchSpace(enable_lr=False, enable_nn=False),
                                     report_render='excel',
                                     report_render_options={'file_path': file_path})
        return experiment

    run_export_excel_report(maker)


def test_disable_cv_render():
    def maker(df_train, target, df_eval, file_path):
        experiment = make_experiment(PlainModel, df_train,
                                     target=target,
                                     eval_data=df_eval,
                                     cv=False,
                                     test_data=df_eval.copy(),
                                     drift_detection_threshold=0.4,
                                     drift_detection_min_features=3,
                                     drift_detection_remove_size=0.5,
                                     search_space=PlainSearchSpace(enable_lr=False, enable_nn=False),
                                     report_render='excel',
                                     report_render_options={'file_path': file_path})
        return experiment

    run_export_excel_report(maker, str_label=False)


def test_report_with_pseudo():
    def maker(df_train, target, df_eval, file_path):
        experiment = make_experiment(PlainModel, df_train,
                                     target=target,
                                     eval_data=df_eval,
                                     cv=False,
                                     test_data=df_eval.copy(),
                                     drift_detection_threshold=0.4,
                                     drift_detection_min_features=3,
                                     drift_detection_remove_size=0.5,
                                     search_space=PlainSearchSpace(enable_lr=False, enable_nn=False),
                                     pseudo_labeling=True,
                                     report_render='excel',
                                     report_render_options={'file_path': file_path})
        return experiment

    experiment_meta = run_export_excel_report(maker, str_label=False)
    assert len(experiment_meta.steps) == 8


def test_obj_render():
    def maker(df_train, target, df_eval, file_path):
        from hypernets.experiment.report import ExcelReportRender
        experiment = make_experiment(PlainModel, df_train,
                                     target=target,
                                     eval_data=df_eval,
                                     test_data=df_eval.copy(),
                                     drift_detection_threshold=0.4,
                                     drift_detection_min_features=3,
                                     drift_detection_remove_size=0.5,
                                     search_space=PlainSearchSpace(enable_lr=False, enable_nn=False),
                                     report_render=ExcelReportRender(file_path))
        return experiment

    run_export_excel_report(maker)


def test_no_eval_data_render():
    def maker(df_train, target, df_eval, file_path):
        experiment = make_experiment(PlainModel, df_train,
                                     target=target,
                                     test_data=df_eval.copy(),
                                     drift_detection_threshold=0.4,
                                     drift_detection_min_features=3,
                                     drift_detection_remove_size=0.5,
                                     search_space=PlainSearchSpace(enable_lr=False, enable_nn=False),
                                     report_render='excel',
                                     report_render_options={'file_path': file_path})
        return experiment

    run_export_excel_report(maker, has_eval_data=False)


def test_regression_task_report():
    df = dsutils.load_boston()
    df['Constant'] = [0 for i in range(df.shape[0])]
    df['Id'] = [i for i in range(df.shape[0])]

    target = 'target'
    df_train, df_eval = train_test_split(df, test_size=0.2)

    df_train['Drifted'] = np.random.random(df_train.shape[0])
    df_eval['Drifted'] = np.random.random(df_eval.shape[0]) * 100
    file_path = common_util.get_temp_file_path(prefix="report_excel_", suffix=".xlsx")
    print(file_path)
    experiment = make_experiment(PlainModel, df_train,
                                 target=target,
                                 eval_data=df_eval.copy(),
                                 test_data=df_eval.copy(),
                                 drift_detection_threshold=0.4,
                                 drift_detection_min_features=3,
                                 drift_detection_remove_size=0.5,
                                 search_space=PlainSearchSpace(enable_lr=False, enable_nn=False,
                                                               enable_dt=False, enable_dtr=True),
                                 report_render='excel',
                                 report_render_options={'file_path': file_path})
    estimator = experiment.run(max_trials=1)
    assert estimator is not None
    mlr_callback = None
    mle_callback = None
    for callback in experiment.callbacks:
        if isinstance(callback, MLReportCallback):
            mlr_callback = callback
        if isinstance(callback, MLEvaluateCallback):
            mle_callback = callback

    assert mlr_callback is not None
    _experiment_meta: ExperimentMeta = mlr_callback.experiment_meta_

    assert len(_experiment_meta.resource_usage) > 0
    assert len(_experiment_meta.steps) == 5
    assert os.path.exists(file_path)

    assert mle_callback is not None
    assert _experiment_meta.evaluation_metrics is not None
    assert len(_experiment_meta.prediction_elapsed) == 2
    assert len(_experiment_meta.datasets) == 3
