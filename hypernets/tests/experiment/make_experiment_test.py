from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import make_experiment
from hypernets.tabular.datasets import dsutils


def test_experiemnt_with_blood():
    df = dsutils.load_blood()
    experiment = make_experiment(PlainModel, df, target='Class', search_space=PlainSearchSpace())
    estimator = experiment.run(max_trials=3)
    print(estimator)
    assert estimator is not None
