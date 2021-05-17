from sklearn.model_selection import train_test_split

from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import GeneralExperiment
from hypernets.searchers import make_searcher
from hypernets.tabular.datasets import dsutils


def create_hyper_model(reward_metric='auc', optimize_direction='max'):
    search_space = PlainSearchSpace()
    searcher = make_searcher('random', search_space_fn=search_space, optimize_direction=optimize_direction)
    hyper_model = PlainModel(searcher=searcher, reward_metric=reward_metric, callbacks=[])

    return hyper_model


def test_general_experiment_of_heart_disease_simple():
    hyper_model = create_hyper_model()

    X = dsutils.load_heart_disease_uci()
    y = X.pop('target')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    experiment = GeneralExperiment(hyper_model, X_train, y_train, eval_size=0.3)
    estimator = experiment.run(max_trials=5)
    trials = hyper_model.get_top_trials(5)

    assert estimator
    assert 1 < len(trials) <= 5

    score = estimator.evaluate(X_test, y_test, metrics=['auc', 'accuracy', 'f1', 'recall', 'precision'])
    print('evaluate score:', score)
    assert score


def test_general_experiment_of_heart_disease_with_eval_and_cv():
    hyper_model = create_hyper_model()

    X = dsutils.load_heart_disease_uci()
    y = X.pop('target')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.3)

    experiment = GeneralExperiment(hyper_model, X_train, y_train, X_eval=X_eval, y_eval=y_eval, X_test=X_test)
    estimator = experiment.run(max_trials=5, cv=True)
    trials = hyper_model.get_top_trials(5)

    assert estimator
    assert 1 < len(trials) <= 5

    score = estimator.evaluate(X_test, y_test, metrics=['auc', 'accuracy', 'f1', 'recall', 'precision'])
    print('evaluate score:', score)
    assert score
