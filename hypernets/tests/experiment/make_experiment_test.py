from sklearn.model_selection import train_test_split

from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
from hypernets.experiment import make_experiment
from hypernets.experiment.compete import StepNames
from hypernets.tabular.datasets import dsutils


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
