from sklearn.metrics import get_scorer

from hypernets.core.callbacks import SummaryCallback
from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import CompeteExperiment
from hypernets.searchers import make_searcher
from hypernets.tabular import dask_ex as dex
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.metrics import calc_score, metric_to_scoring
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from hypernets.tests.tabular.dask_transofromer_test import setup_dask


class DaskPlainModel(PlainModel):
    def _get_estimator(self, space_sample):
        estimator = super()._get_estimator(space_sample)

        return dex.wrap_local_estimator(estimator)


def create_hyper_model(reward_metric='auc', optimize_direction='max', with_dask=False):
    search_space = PlainSearchSpace(enable_dt=True, enable_lr=True, enable_nn=False)
    searcher = make_searcher('random', search_space_fn=search_space, optimize_direction=optimize_direction)
    if with_dask:
        cls = DaskPlainModel
    else:
        cls = PlainModel
    hyper_model = cls(searcher=searcher, reward_metric=reward_metric, callbacks=[SummaryCallback()])

    return hyper_model


def experiment_with_bank_data(init_kwargs, run_kwargs, row_count=3000, with_dask=False):
    hyper_model = create_hyper_model(with_dask=with_dask)
    scorer = get_scorer(metric_to_scoring(hyper_model.reward_metric))
    X = dsutils.load_bank()
    if row_count is not None:
        X = X.head(row_count)
    X = MultiLabelEncoder().fit_transform(X)

    if with_dask:
        setup_dask(None)
        X = dex.dd.from_pandas(X, npartitions=1)

    y = X.pop('y')

    X_train, X_test, y_train, y_test = \
        dex.train_test_split(X, y, test_size=0.3, random_state=9527)
    X_train, X_eval, y_train, y_eval = \
        dex.train_test_split(X_train, y_train, test_size=0.3, random_state=9527)

    init_kwargs = {
        'X_eval': X_eval, 'y_eval': y_eval, 'X_test': X_test,
        'scorer': scorer,
        'ensemble_size': 0,
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
        preds, proba = dex.compute(preds, proba)

    score = calc_score(y_test, preds, proba, metrics=['auc', 'accuracy', 'f1', 'recall', 'precision'])
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


def test_without_dd():
    experiment_with_bank_data(dict(drift_detection=False), {})


def test_with_cd():
    experiment_with_bank_data(dict(collinearity_detection=True), {})


def test_with_pl():
    experiment_with_bank_data(dict(drift_detection=False, pseudo_labeling=True), {})


def test_with_pi():
    experiment_with_bank_data(dict(drift_detection=False,
                                   feature_reselection=True,
                                   feature_reselection_threshold=0.001), {})


def test_with_pl_dask():
    experiment_with_bank_data(dict(cv=False, drift_detection=False, pseudo_labeling=True), {},
                              with_dask=True)


def test_with_ensemble_dask():
    experiment_with_bank_data(dict(ensemble_size=5, cv=False, drift_detection_num_folds=3), {},
                              with_dask=True)


def test_with_cv_ensemble2_dask():
    experiment_with_bank_data(dict(ensemble_size=5, cv=True, drift_detection=False), {},
                              row_count=6000, with_dask=True)


def test_with_cv_ensemble_dask():
    experiment_with_bank_data(dict(ensemble_size=5, cv=True, drift_detection=False), {},
                              row_count=6000, with_dask=True)
