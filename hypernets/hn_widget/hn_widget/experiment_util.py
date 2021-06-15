import json, copy

class StepType:
    DataCleaning = 'DataCleanStep'
    CollinearityDetection = 'MulticollinearityDetectStep'
    DriftDetection =  'DriftDetectStep'
    SpaceSearch = 'SpaceSearchStep'
    FeatureSelection = 'FeatureImportanceSelectionStep'
    PsudoLabeling  = 'PseudoLabelStep'
    ReSpaceSearch = 'ReSpaceSearch'
    Ensemble = 'EnsembleStep'

class StepStatus:
  Wait = 'wait'
  Process = 'process'
  Finish = 'finish'
  Error = 'error'


class StepData:

    def __init__(self, index, name,  type, status, configuration, extension, start_datetime, end_datetime):
        self.index = index
        self.name = name
        self.type = type
        self.status = status
        self.configuration = configuration
        self.extension = extension
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

    def to_dict(self):
        return {
            'index': self.index,
            'name': self.name,
            'type': self.type,
            'status': self.status,
            'configuration': self.configuration,
            'extension': self.extension,
            'start_datetime': self.start_datetime,
            'end_datetime': self.end_datetime
        }

    def to_json(self):
        return json.dumps(self.to_dict())


def get_extra_attr(obj, name, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    else:
        return None

def get_step_status(step):
    return get_extra_attr(step, 'status', default=StepStatus.Wait)

def get_step_index(experiment, step_name):
    for i, step in enumerate(experiment.steps):
        if step.name == step_name:
            return i
    return -1

def get_step_kind(self, step):
    class_name = step.__class__.__name__
    return class_name

#
# for col, score in scores.items():
#     if score <= variable_shift_threshold:
#         remain_features.append(col)
#     else:
#         remove_features.append(col)
#         logger.info(f'Remove shift variables:{col},  score:{score}')
#

def extract_drift_step(config, extension, step):
    import copy
    extension = copy.deepcopy(extension)

    if get_step_status(step) != StepStatus.Finish:
        return extension
    extension['drifted_features_auc'] = []
    if 'scores' in extension and extension['scores'] is not None:
        scores = extension['scores']
        variable_shift_threshold = config['variable_shift_threshold']
        if config['remove_shift_variable']:
            remove_features = []
            for col, score in scores.items():
                if score <= variable_shift_threshold:
                    pass
                else:
                    remove_features.append({'feature': col, 'score': score})
            remove_features = sorted(remove_features, key=lambda item: item['score'], reverse=True)
            extension['drifted_features_auc'] = remove_features

    def get_importance(col, feature_names, feature_importances):
        for i, c in enumerate(feature_names):
            if c == col:
                return feature_importances[i]
        return None

    historys = extension['history']
    if historys is not None and len(historys) > 0:
        removed_features_in_epochs = []
        for i, history in enumerate(historys):
            feature_names = history['feature_names']
            feature_importances = history['feature_importances']

            removed_features = [] if 'removed_features' not in history else history['removed_features']
            removed_features_importances = [{'feature': f, 'importance': get_importance(f, feature_names, feature_importances) }  for f in feature_names]
            removed_features_importances = sorted(removed_features_importances, key=lambda item: item['importance'])

            d = {
                "epoch": i,
                "elapsed": history['elapsed'],
                "removed_features": removed_features_importances
            }

            removed_features_in_epochs.append(d)
        extension['removed_features_in_epochs'] = removed_features_in_epochs

    del extension['scores']
    del extension['history']

    return extension


def extract_multi_linearity_step(step, extension):
    import copy
    extension = copy.deepcopy(extension)

    if get_step_status(step) != StepStatus.Finish:
        return extension
    feature_clusters = extension['feature_clusters']
    unselected_features = []
    for fs in feature_clusters:
        if len(fs) > 1:
            reserved = fs.copy().pop(0)
            for f_i, remove in enumerate(fs):
                if f_i > 0:  # drop first element
                    unselected_features.append({"removed": remove, "reserved": reserved})
    output_extension = {}
    output_extension['unselected_features'] = unselected_features
    return output_extension


def extract_step(index, step):
    stepType = step.__class__.__name__

    configuration = copy.deepcopy(step.get_params())
    del configuration['name']  # Ignore name
    extension = step.get_fitted_params()
    _status = get_extra_attr(step, 'status')
    status = StepStatus.Wait if _status is None else _status

    if stepType == StepType.Ensemble:
        configuration['scorer'] = None
    elif stepType == StepType.SpaceSearch:
        extension['history'] = None
    elif stepType == StepType.CollinearityDetection:
        extension = extract_multi_linearity_step(step, extension)
    elif stepType == StepType.DriftDetection:
        extension = extract_drift_step(configuration, extension, step)
    elif stepType == StepType.DataCleaning:
        pass
        # extension['unselected_features'] = extension['unselected_reason']  # fixme
        # status = 'finish'
    elif stepType == StepType.Ensemble:
        configuration['scorer'] = None
        extension['estimator'] = None
    else:
        pass

    d = \
        StepData(index=index,
                 name=step.name,
                 type=step.__class__.__name__,
                 status=status,
                 configuration=configuration,
                 extension=extension,
                 start_datetime=get_extra_attr(step, 'start_datetime'),
                 end_datetime=get_extra_attr(step, 'end_datetime'))
    return d.to_dict()


def extract_experiment(compete_experiment):
    step_dict_list = []
    for i, step in enumerate(compete_experiment.steps):
        step_dict_list.append(extract_step(i, step))
    return {"steps": step_dict_list}


