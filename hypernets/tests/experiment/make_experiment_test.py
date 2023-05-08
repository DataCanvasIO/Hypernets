import os.path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import make_experiment, MLEvaluateCallback, MLReportCallback, ExperimentMeta
from hypernets.experiment.compete import StepNames
from hypernets.tabular import get_tool_box
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.feature_generators import is_feature_generator_ready
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from hypernets.utils import common as common_util
from hypernets.searchers.nsga_searcher import NSGAIISearcher


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
                                 feature_generation=is_feature_generator_ready,
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


class CatPlainModel(PlainModel):

    def __init__(self, searcher, dispatcher=None, callbacks=None, reward_metric=None, task=None,
                 discriminator=None):
        super(CatPlainModel, self).__init__(searcher, dispatcher=dispatcher, callbacks=callbacks,
                                            reward_metric=reward_metric, task=task)
        self.transformer = MultiLabelEncoder


class TestMOOExperiment:

    @classmethod
    def setup_class(cls):
        df = dsutils.load_bank().head(1000)
        df['y'] = LabelEncoder().fit_transform(df['y'])
        tb = get_tool_box(df)
        df_train, df_test = tb.train_test_split(df, test_size=0.3, random_state=9527)
        cls.df_train = df_train
        cls.df_test = df_test

    def check_exp(self, experiment, estimators):
        assert estimators is not None
        assert isinstance(estimators, list)
        hyper_model = experiment.hyper_model_
        estimator = estimators[0]
        searcher = experiment.hyper_model_.searcher
        assert searcher.get_best()
        # fig, ax = hyper_model.history.plot_best_trials()
        # assert fig is not None
        # assert ax is not None
        # fig, ax = hyper_model.searcher.plot_population()
        # assert fig is not None
        # assert ax is not None
        optimal_set = searcher.get_nondominated_set()
        assert optimal_set is not None
        # assert optimal_set[0].scores[1] > 0
        df_trials = hyper_model.history.to_df().copy().drop(['scores', 'reward'], axis=1)
        print(df_trials[df_trials['non_dominated'] == True])
        df_test = self.df_test.copy()
        X_test = df_test.copy()
        y_test = X_test.pop('y')
        preds = estimator.predict(X_test)
        proba = estimator.predict_proba(X_test)
        tb = get_tool_box(df_test)
        score = tb.metrics.calc_score(y_test, preds, proba, metrics=['auc', 'accuracy', 'f1', 'recall', 'precision'])
        print('evaluate score:', score)
        assert score

    def test_nsga2(self):
        df_train = self.df_train.copy()
        df_test = self.df_test.copy()
        experiment = make_experiment(CatPlainModel, df_train,
                                     eval_data=df_test,
                                     callbacks=[],
                                     random_state=1234,
                                     search_callbacks=[],
                                     target='y',
                                     searcher='nsga2',  # available MOO searcher: moead, nsga2, rnsga2
                                     searcher_options={'population_size': 5},
                                     reward_metric='logloss',
                                     objectives=['nf'],
                                     drift_detection=False,
                                     early_stopping_rounds=10,
                                     search_space=PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True))

        estimators = experiment.run(max_trials=10)
        self.check_exp(experiment, estimators)

    def test_nsga2_psi(self):
        df_train = self.df_train.copy()
        df_test = self.df_test.copy()
        X_test = df_test.copy().drop('y', axis=1)
        experiment = make_experiment(CatPlainModel, df_train,
                                     eval_data=df_test,
                                     test_data=X_test,
                                     callbacks=[],
                                     random_state=1234,
                                     search_callbacks=[],
                                     target='y',
                                     searcher='nsga2',  # available MOO searcher: moead, nsga2, rnsga2
                                     searcher_options={'population_size': 5},
                                     reward_metric='auc',
                                     objectives=['psi'],
                                     drift_detection=False,
                                     early_stopping_rounds=10,
                                     search_space=PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True))

        estimators = experiment.run(max_trials=10)
        self.check_exp(experiment, estimators)

    def test_rnsga2(self):
        df_train = self.df_train.copy()
        df_test = self.df_test.copy()
        experiment = make_experiment(CatPlainModel, df_train,
                                     eval_data=df_test.copy(),
                                     callbacks=[],
                                     random_state=1234,
                                     search_callbacks=[],
                                     target='y',
                                     searcher='rnsga2',  # available MOO searchers: moead, nsga2, rnsga2
                                     searcher_options=dict(ref_point=np.array([0.1, 2]), weights=np.array([0.1, 2]),
                                                           population_size=5),
                                     reward_metric='logloss',
                                     objectives=['nf'],
                                     early_stopping_rounds=10,
                                     drift_detection=False,
                                     search_space=PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True))

        estimators = experiment.run(max_trials=10)
        self.check_exp(experiment, estimators)

    def test_moead(self):
        df_train = self.df_train.copy()
        df_test = self.df_test.copy()
        experiment = make_experiment(CatPlainModel, df_train,
                                     eval_data=df_test.copy(),
                                     callbacks=[],
                                     random_state=1234,
                                     search_callbacks=[],
                                     target='y',
                                     searcher='moead',  # available MOO searcher: moead, nsga2, rnsga2
                                     reward_metric='logloss',
                                     objectives=['nf'],
                                     drift_detection=False,
                                     early_stopping_rounds=10,
                                     search_space=PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True))

        estimators = experiment.run(max_trials=10)
        self.check_exp(experiment, estimators)

