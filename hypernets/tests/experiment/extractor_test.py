from hypernets.tests.experiment import experiment_factory
from hypernets.experiment.compete import DataCleanStep, DriftDetectStep
from hypernets.experiment import ExperimentExtractor, StepMeta
import pytest
from hypernets.tabular.feature_generators import is_feature_generator_ready


def _run_experiment(creator):
    exp = creator()
    estimator = exp.run(max_trials=3)
    experiment_data = ExperimentExtractor(exp).extract()
    assert estimator is not None
    return experiment_data,  estimator


def test_data_clean_extractor():
    exp_data, estimator = _run_experiment(experiment_factory.create_data_clean_experiment)
    step = exp_data.steps[1]
    extension = step.extension

    assert step.type == DataCleanStep.__name__
    unselected_reason = extension['unselected_reason']
    unselected_features = unselected_reason.keys()
    assert len(unselected_features) == 2
    assert 'Constant' in unselected_features
    assert 'Id' in unselected_features
    assert unselected_reason['Constant'] == 'constant'
    assert unselected_reason['Id'] == 'idness'


def test_drift_detect_extractor():
    exp_data, estimator = _run_experiment(experiment_factory.create_drift_detection_experiment)
    dd_step = exp_data.steps[2]
    unselected_features = dd_step.extension['unselected_features']
    assert "Drifted" == unselected_features['over_variable_threshold'][0][0]

    over_threshold_feature_epochs = unselected_features['over_threshold']
    if len(over_threshold_feature_epochs) > 0:
        over_threshold_feature_epoch = over_threshold_feature_epochs[0]
        assert over_threshold_feature_epoch['elapsed']
        assert over_threshold_feature_epoch['epoch'] == 0
        assert isinstance(over_threshold_feature_epoch['removed_features'][0][0], str)


def run_feature_selection_extractor(creator, fs_step_index):
    exp_data, estimator = _run_experiment(creator)
    fe_step = exp_data.steps[fs_step_index]
    feature_importances = fe_step.extension['importances']
    feature_names = [f['name'] for f in feature_importances]

    feature = feature_importances[0]
    assert feature['importance'] > 0

    assert "LSTAT" in feature_names
    assert feature['dropped'] is False
    assert feature_importances[-1]['dropped'] is True  # lowest importance feature


def test_feature_selection_extractor():
    run_feature_selection_extractor(experiment_factory.create_feature_selection_experiment, 2)


def test_feature_reselection_extractor():
    run_feature_selection_extractor(experiment_factory.create_feature_reselection_experiment, 3)


def test_multicollinearity_detect_extractor():
    exp_data, estimator = _run_experiment(experiment_factory.create_multicollinearity_detect_experiment)
    multicollinearity_detect_step = exp_data.steps[2]
    print(multicollinearity_detect_step)
    unselected_features = multicollinearity_detect_step.extension['unselected_features']
    assert len(unselected_features.keys()) > 0
    assert "INDUS" in unselected_features
    assert unselected_features['INDUS']['reserved'] == 'CRIM'


@pytest.mark.skipif(not is_feature_generator_ready, reason='feature_generator is not ready')
def test_feature_generation_extractor():
    exp_data, estimator = _run_experiment(experiment_factory.create_feature_generation_experiment)
    fg_step = exp_data.steps[2]

    output_features = fg_step.extension['outputFeatures']
    assert len(output_features) > 0

    fg_feature = output_features[0]

    assert fg_feature['name']
    assert fg_feature['primitive']
    assert 'parentFeatures' in fg_feature
    assert fg_feature['variableType']
    assert fg_feature['derivationType']

    output_feature_names = [of['name'] for of in output_features]
    assert "AGE__A__B" in output_feature_names
    assert "RAD" in output_feature_names
    assert "RM__S__ZN" in output_feature_names


def test_pseudo_labeling_extractor():
    exp_data, estimator = _run_experiment(experiment_factory.create_pseudo_labeling_experiment)
    pl_step = exp_data.steps[4]

    samples = pl_step.extension['samples']
    assert samples[0] >= 0
    assert samples[1] >= 0

    probability_density = pl_step.extension['probabilityDensity']
    assert probability_density[0]['gaussian']
    assert probability_density[1]['gaussian']

    assert len(probability_density[0]['gaussian']['X']) > 0
    assert len(probability_density[0]['gaussian']['probaDensity']) > 0
