from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import make_experiment
from hypernets.experiment.compete import StepNames
from hypernets.tabular.datasets import dsutils


def test_experiemnt_with_blood_simple():
    df = dsutils.load_blood()
    experiment = make_experiment(PlainModel, df, target='Class', search_space=PlainSearchSpace())
    estimator = experiment.run(max_trials=3)
    print(estimator)
    assert estimator is not None


def test_experiemnt_with_blood_downsample():
    df = dsutils.load_blood()
    experiment = make_experiment(PlainModel, df, target='Class', search_space=PlainSearchSpace(),
                                 down_sample_search=True, down_sample_search_frac=0.1,
                                 # log_level='info',
                                 )
    estimator = experiment.run(max_trials=3)
    print(estimator)
    assert estimator is not None


def test_experiemnt_with_blood_full_features():
    df = dsutils.load_blood()
    experiment = make_experiment(PlainModel, df, target='Class', search_space=PlainSearchSpace(),
                                 feature_generation=True,
                                 collinearity_detection=True,
                                 drift_detection=True,
                                 feature_selection=True,
                                 down_sample_search=True, down_sample_search_frac=0.2,
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
