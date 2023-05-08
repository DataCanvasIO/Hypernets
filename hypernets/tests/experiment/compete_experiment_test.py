from datetime import datetime

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from hypernets.core import SummaryCallback
from hypernets.core.objective import Objective
from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import CompeteExperiment
from hypernets.model.objectives import PredictionObjective
from hypernets.searchers.nsga_searcher import NSGAIISearcher
from hypernets.tabular import get_tool_box
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.feature_generators import is_feature_generator_ready
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from hypernets.tests.model.plain_model_test import create_plain_model
from hypernets.tests.tabular.tb_dask import if_dask_ready, is_dask_installed, setup_dask
from hypernets.utils import const

if is_dask_installed:
    import dask.dataframe as dd


def experiment_with_bank_data(init_kwargs, run_kwargs, row_count=3000, with_dask=False):
    hyper_model = create_plain_model(with_encoder=True, with_dask=with_dask)
    X = dsutils.load_bank()
    if row_count is not None:
        X = X.head(row_count)
    X['y'] = LabelEncoder().fit_transform(X['y'])

    if with_dask:
        setup_dask(None)
        X = dd.from_pandas(X, npartitions=1)

    y = X.pop('y')

    tb = get_tool_box(X, y)
    scorer = tb.metrics.metric_to_scoring(hyper_model.reward_metric)

    X_train, X_test, y_train, y_test = \
        tb.train_test_split(X, y, test_size=0.3, random_state=9527)
    X_train, X_eval, y_train, y_eval = \
        tb.train_test_split(X_train, y_train, test_size=0.3, random_state=9527)

    init_kwargs = {
        'X_eval': X_eval, 'y_eval': y_eval, 'X_test': X_test,
        'scorer': scorer,
        'ensemble_size': 0,
        'drift_detection': False,
        **init_kwargs
    }
    run_kwargs = {
        'max_trials': 3,
        **run_kwargs
    }
    experiment = CompeteExperiment(hyper_model, X_train, y_train, **init_kwargs)
    estimator = experiment.run(**run_kwargs)

    assert estimator

    preds = estimator.predict(X_test)
    proba = estimator.predict_proba(X_test)

    if with_dask:
        preds, proba = tb.to_local(preds, proba)

    score = tb.metrics.calc_score(y_test, preds, proba, metrics=['auc', 'accuracy', 'f1', 'recall', 'precision'])
    print('evaluate score:', score)
    assert score


def experiment_with_movie_lens(init_kwargs, run_kwargs, row_count=None, with_dask=False):
    hyper_model = create_plain_model(reward_metric='f1', with_encoder=True, with_dask=with_dask)

    X = dsutils.load_movielens()
    # X['genres'] = X['genres'].apply(lambda s: s.replace('|', ' '))
    X['timestamp'] = X['timestamp'].apply(datetime.fromtimestamp)
    if row_count is not None:
        X = X.head(row_count)

    if with_dask:
        setup_dask(None)
        X = dd.from_pandas(X, npartitions=1)

    y = X.pop('rating')

    tb = get_tool_box(X, y)

    X_train, X_test, y_train, y_test = \
        tb.train_test_split(X, y, test_size=0.3, random_state=9527)
    X_train, X_eval, y_train, y_eval = \
        tb.train_test_split(X_train, y_train, test_size=0.3, random_state=9527)

    init_kwargs = {
        'X_eval': X_eval, 'y_eval': y_eval, 'X_test': X_test,
        'ensemble_size': 0,
        'drift_detection': False,
        **init_kwargs
    }
    run_kwargs = {
        'max_trials': 3,
        **run_kwargs
    }
    experiment = CompeteExperiment(hyper_model, X_train, y_train, **init_kwargs)
    estimator = experiment.run(**run_kwargs)

    assert estimator

    preds = estimator.predict(X_test)
    proba = estimator.predict_proba(X_test)

    if with_dask:
        preds, proba = tb.to_local(preds, proba)

    score = tb.metrics.calc_score(y_test, preds, proba,
                                  metrics=['auc', 'accuracy', 'f1', 'recall', 'precision'],
                                  task=experiment.task)
    print('evaluate score:', score)
    assert score


def test_simple():
    experiment_with_bank_data({}, {})


def test_without_eval():
    experiment_with_bank_data(dict(X_eval=None, y_eval=None, ), {})


def test_without_xtest():
    experiment_with_bank_data(dict(X_test=None), {})


def test_without_eval_xtest():
    experiment_with_bank_data(dict(X_eval=None, y_eval=None, X_test=None), {})


def test_with_adversarial_validation():
    experiment_with_bank_data(dict(cv=False,
                                   X_eval=None,
                                   y_eval=None,
                                   train_test_split_strategy='adversarial_validation'), {})


def test_with_ensemble():
    experiment_with_bank_data(dict(ensemble_size=5), {})


def test_without_cv():
    experiment_with_bank_data(dict(cv=False), {})


@pytest.mark.skipif(not is_feature_generator_ready, reason='feature_generator is not ready')
def test_with_feature_generation():
    experiment_with_movie_lens(dict(feature_generation=True,
                                    feature_generation_text_cols=['title']), {})


def test_with_dd():
    experiment_with_bank_data(dict(drift_detection=True), {})


def test_with_cd():
    experiment_with_bank_data(dict(collinearity_detection=True), {})


def test_with_fi_threshold():
    experiment_with_bank_data(dict(feature_selection=True,
                                   feature_selection_threshold=0.0000001), {})


def test_with_fi_quantile():
    experiment_with_bank_data(dict(feature_selection=True,
                                   feature_selection_strategy='quantile',
                                   feature_selection_quantile=0.4), {})


def test_with_fi_number():
    experiment_with_bank_data(dict(feature_selection=True,
                                   feature_selection_strategy='number',
                                   feature_selection_number=10), {})


def test_with_pl_threshold():
    experiment_with_bank_data(dict(pseudo_labeling=True), {})


def test_with_pl_quantile():
    experiment_with_bank_data(dict(pseudo_labeling=True,
                                   pseudo_labeling_strategy='quantile'), {})


def test_with_pl_number():
    experiment_with_bank_data(dict(drift_detection=False, pseudo_labeling=True,
                                   pseudo_labeling_strategy='number'), {})


def test_with_pi():
    experiment_with_bank_data(dict(feature_reselection=True,
                                   feature_reselection_threshold=0.0001), {})


@pytest.mark.skipif(not is_feature_generator_ready, reason='feature_generator is not ready')
def test_with_feature_generation_and_selection():
    experiment_with_movie_lens(dict(feature_generation=True, feature_selection=True,
                                    feature_generation_text_cols=['title']), {})


@if_dask_ready
def test_with_pl_dask():
    experiment_with_bank_data(dict(cv=False, pseudo_labeling=True), {},
                              with_dask=True)


@if_dask_ready
def test_with_ensemble_dask():
    experiment_with_bank_data(dict(ensemble_size=5, cv=False), {},
                              with_dask=True)


@if_dask_ready
def test_with_cv_ensemble_dask():
    experiment_with_bank_data(dict(ensemble_size=5, cv=True), {},
                              row_count=6000, with_dask=True)


@if_dask_ready
@pytest.mark.skipif(not is_feature_generator_ready, reason='feature_generator is not ready')
def test_with_feature_generator_dask():
    experiment_with_movie_lens(dict(feature_generation=True, feature_selection=True,
                                    feature_generation_text_cols=['title']), {}, with_dask=True)


class PlainContextObjective(Objective):

    def __init__(self):
        super(PlainContextObjective, self).__init__('plain_context', 'min')

    def _evaluate(self, trial, estimator, X_train, y_train, X_val, y_val, X_test=None, **kwargs) -> float:
        exp = trial.context.get('exp')
        assert exp is not None and isinstance(exp, CompeteExperiment)  # get experiment in Objective
        return np.random.random()

    def _evaluate_cv(self, trial, estimator, X_trains, y_trains, X_vals, y_vals, X_test=None, **kwargs) -> float:
        exp = trial.context.get('exp')
        assert exp is not None and isinstance(exp, CompeteExperiment)  # get experiment in Objective
        return np.random.random()


def test_moo_context():
    search_space = PlainSearchSpace(enable_dt=True, enable_lr=False, enable_nn=True)
    rs = NSGAIISearcher(search_space, objectives=[PredictionObjective.create("auc", task=const.TASK_BINARY),
                                                  PlainContextObjective()],
                        population_size=10)

    hyper_model = PlainModel(rs, task='binary', callbacks=[SummaryCallback()], transformer=MultiLabelEncoder)

    X = dsutils.load_bank().sample(1000)
    X['y'] = LabelEncoder().fit_transform(X['y'])
    y = X.pop('y')

    tb = get_tool_box(X, y)

    X_train, X_test, y_train, y_test = \
        tb.train_test_split(X, y, test_size=0.3, random_state=9527)
    X_train, X_eval, y_train, y_eval = \
        tb.train_test_split(X_train, y_train, test_size=0.3, random_state=9527)

    init_kwargs = {
        'X_eval': X_eval, 'y_eval': y_eval, 'X_test': X_test,
        'ensemble_size': 0,
        'drift_detection': False,
    }
    run_kwargs = {
        'max_trials': 3,
    }
    from hypernets.tabular.metrics import metric_to_scoring
    experiment = CompeteExperiment(hyper_model, X_train, y_train, scorer=metric_to_scoring("auc"), **init_kwargs)

    estimators = experiment.run(**run_kwargs)

    assert estimators
    assert isinstance(estimators, list)

    estimator = estimators[0]

    optimal_set = experiment.hyper_model_.searcher.get_nondominated_set()
    assert experiment.hyper_model_.searcher.get_best()

    assert optimal_set[0].scores[1] > 0

    preds = estimator.predict(X_test)
    proba = estimator.predict_proba(X_test)

    score = tb.metrics.calc_score(y_test, preds, proba, metrics=['auc', 'accuracy', 'f1', 'recall', 'precision'])
    print('evaluate score:', score)
    assert score
