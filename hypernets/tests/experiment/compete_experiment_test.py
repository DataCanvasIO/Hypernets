from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer

from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import CompeteExperiment
from hypernets.searchers import make_searcher
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.sklearn_ex import MultiLabelEncoder
from hypernets.core.callbacks import SummaryCallback
from hypernets.tabular.metrics import calc_score, metric_to_scoring


def create_hyper_model(reward_metric='auc', optimize_direction='max'):
    search_space = PlainSearchSpace()
    searcher = make_searcher('random', search_space_fn=search_space, optimize_direction=optimize_direction)
    hyper_model = PlainModel(searcher=searcher, reward_metric=reward_metric, callbacks=[SummaryCallback()])

    return hyper_model


def run_compete_experiment_with_bank_data(init_kwargs, run_kwargs):
    hyper_model = create_hyper_model()
    scorer = get_scorer(metric_to_scoring(hyper_model.reward_metric))
    X = dsutils.load_bank().head(3000)
    X = MultiLabelEncoder().fit_transform(X)
    y = X.pop('y')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.3)

    init_kwargs = {
        'X_eval': X_eval, 'y_eval': y_eval, 'X_test': X_test,
        'scorer': scorer,
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

    score = calc_score(y_test, preds, proba, metrics=['auc', 'accuracy', 'f1', 'recall', 'precision'])
    print('evaluate score:', score)
    assert score


def test_simple():
    run_compete_experiment_with_bank_data({}, {})


def test_without_cv():
    run_compete_experiment_with_bank_data(dict(cv=False), {})


def test_without_dd():
    run_compete_experiment_with_bank_data(dict(drift_detection=False), {})


def test_with_pl():
    run_compete_experiment_with_bank_data(dict(drift_detection=False, pseudo_labeling=True), {})


def test_with_pi():
    run_compete_experiment_with_bank_data(dict(drift_detection=False,
                                               feature_reselection=True,
                                               feature_reselection_threshold=0.001), {})
