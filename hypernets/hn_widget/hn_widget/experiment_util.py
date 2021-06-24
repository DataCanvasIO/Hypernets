import json, copy
import numpy as np
import copy


class StepType:
    DataCleaning = 'DataCleanStep'
    CollinearityDetection = 'MulticollinearityDetectStep'
    DriftDetection =  'DriftDetectStep'
    SpaceSearch = 'SpaceSearchStep'
    FeatureSelection = 'FeatureImportanceSelectionStep'
    PsudoLabeling  = 'PseudoLabelStep'
    FeatureGeneration  = 'FeatureGenerationStep'
    PermutationImportanceSelection  = 'PermutationImportanceSelectionStep'
    ReSpaceSearch = 'ReSpaceSearch'
    Ensemble = 'EnsembleStep'
    FinalTrain = 'FinalTrainStep'

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


class Extractor:

    def __init__(self, step):
        self.step = step

    def get_configuration(self):
        configuration = copy.deepcopy(self.step.get_params())
        return configuration

    def get_extension(self):
        if get_step_status(self.step) != StepStatus.Finish:
            return {}
        else:
            extension = copy.deepcopy(self.step.get_fitted_params())
            return self.handle_extenion(extension)

    def handle_extenion(self, extension):
        return extension

class   extract_feature_generation_step(Extractor):

    def handle_extenion(self, extension):
        def get_feature_detail(f):
            return {
                'name': f.get_name(),
                'primitive': type(f.primitive).__name__,
                'parentFeatures': list(map(lambda x: x.get_name(), f.base_features)),
                'variableType': f.variable_type.type_string,
                'derivationType': type(f).__name__
            }
        feature_defs = self.step.transformer_.feature_defs_
        output_features = list(map(lambda f: get_feature_detail(f), feature_defs))
        extension = {"outputFeatures": output_features }
        return extension

class extract_drift_step(Extractor):
    def handle_extenion(self, extension):
        config = super(extract_drift_step, self).get_configuration()
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
            return 0
        historys = extension['history']
        if historys is not None and len(historys) > 0:
            removed_features_in_epochs = []
            for i, history in enumerate(historys):
                feature_names = history['feature_names']
                feature_importances = history['feature_importances'].tolist()

                removed_features = [] if 'removed_features' not in history else history['removed_features']
                removed_features_importances = [{'feature': f, 'importance': get_importance(f, feature_names, feature_importances)} for f in removed_features]
                removed_features_importances = sorted(removed_features_importances, key=lambda item: item['importance'], reverse=True)

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

class abs_feature_selection_step(Extractor):

    def build_importances_result_(self, columns, importances_data, selected_features):
        features = []
        for col, imp in zip(columns, importances_data):
            features.append({
                'name': col,
                'importance': imp,
                'dropped': col not in selected_features
            })
        extension = {
            'importances': sorted(features, key=lambda v: v['importance'], reverse=True)
        }
        return extension

class extract_feature_selection_step(abs_feature_selection_step):
    def handle_extenion(self, extension):
        imps = extension['importances']
        extension['importances'] = imps.tolist() if imps is not None else []
        return self.build_importances_result_(self.step.input_features_, imps, self.step.selected_features_)

class extract_multi_linearity_step(Extractor):

    def handle_extenion(self, extension):
        feature_clusters = extension['feature_clusters']
        unselected_features = []
        for fs in feature_clusters:
            if len(fs) > 1:
                reserved = fs.copy().pop(0)
                for f_i, remove in enumerate(fs):
                    if f_i > 0:  # drop first element
                        unselected_features.append({"removed": remove, "reserved": reserved})
        output_extension = {'unselected_features': unselected_features}
        return output_extension

class extract_psedudo_step(Extractor):
    def get_configuration(self):
        configuration = super(extract_psedudo_step, self).get_configuration()
        del configuration['estimator_builder']
        del configuration['estimator_builder__scorer']
        del configuration['name']
        return configuration

    def handle_extenion(self, extension):
        # step.estimator_builder.estimator_.classes_
        # step.test_proba_
        # step.pseudo_label_stat_
        return configuration, extension

class extract_permutation_importance_step(abs_feature_selection_step):
    def get_configuration(self):
        configuration = super(extract_permutation_importance_step, self).get_configuration()
        configuration['scorer'] = str(configuration['scorer'])
        return configuration

    def handle_extenion(self, extension):
        selected_features = self.step.selected_features_ if self.step.selected_features_  is not None else []
        importances = self.step.importances_
        columns = importances.columns if importances.columns is not None else []
        importances_data = importances.importances_mean.tolist() if importances.importances_mean is not None else []

        return self.build_importances_result_(columns, importances_data, selected_features)

class extract_space_search_step(Extractor):
    def handle_extenion(self, extension):
        extension['history'] = None
        return extension

class extract_final_train_step(Extractor):
    def handle_extenion(self, extension):
        extension = {"estimator": self.step.estimator_.gbm_model.__class__.__name__}
        return extension

class extract_ensemble_step(Extractor):

    def get_configuration(self):
        configuration = super(extract_ensemble_step, self).get_configuration()
        configuration['scorer'] = None
        return configuration

    def handle_extenion(self, extension):
        ensemble = extension['estimator']
        return {
            'weights': np.array(ensemble.weights_).tolist(),
            'scores': np.array(ensemble.scores_).tolist(),
        }

class extract_proba_density(Extractor):

    def get_proba_density_estimation(self, y_proba_on_test, classes):
        from sklearn.neighbors import KernelDensity
        total_class = len(classes)
        total_proba = np.size(y_proba_on_test, 0)

        true_density = [[0]*501 for _ in range(total_class)]
        X_plot_true_density = np.linspace(0, 1, 501)[:, np.newaxis]
        X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]

        probability_density = {}
        for i in range(total_class):
            aclass = classes[i]
            probability_density[str(aclass)] = {}
        
            # calculate the true density
            proba_list = y_proba_on_test[:, i]
            probability_density[str(aclass)]['nSamples'] = len(proba_list)
            for proba in proba_list:
                true_density[i][int(proba*500)] += 1
            probability_density[str(aclass)]['trueDensity'] = {}
            probability_density[str(aclass)]['trueDensity']['X'] = X_plot_true_density
            probability_density[str(aclass)]['trueDensity']['probaDensity'] = list(map(lambda x: x / total_proba, true_density[i]))

            # calculate the gaussian/tophat/epanechnikov density estimation
            proba_list_2d = y_proba_on_test[:, i][:, np.newaxis]
            kernels = ['gaussian', 'tophat', 'epanechnikov']
            for kernel in kernels:
                kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(proba_list_2d)
                log_dens = kde.score_samples(X_plot)
                probability_density[str(aclass)][str(kernel)] = {}
                probability_density[str(aclass)][str(kernel)]['X'] = X_plot
                probability_density[str(aclass)][str(kernel)]['probaDensity'] = np.exp(log_dens)

        return probability_density

def extract_step(index, step):
    stepType = step.__class__.__name__
    extractors = {
        # StepType.DataCleaning: None,
        StepType.FeatureGeneration: extract_feature_generation_step,
        StepType.DriftDetection: extract_drift_step,
        StepType.FeatureSelection: extract_feature_selection_step,
        StepType.CollinearityDetection: extract_multi_linearity_step,
        StepType.PsudoLabeling: extract_psedudo_step,
        StepType.PermutationImportanceSelection: extract_permutation_importance_step,
        StepType.SpaceSearch: extract_space_search_step,
        StepType.FinalTrain: extract_final_train_step,
        StepType.Ensemble: extract_ensemble_step
    }

    extractor = extractors.get(stepType,  Extractor)(step)
    configuration = extractor.get_configuration()
    extension = extractor.get_extension()

    d = \
        StepData(index=index,
                 name=step.name,
                 type=step.__class__.__name__,
                 status=get_step_status(step),
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
