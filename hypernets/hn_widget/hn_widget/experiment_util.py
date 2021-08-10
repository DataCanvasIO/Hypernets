import json, copy
import numpy as np
import copy

import pandas as pd
import numpy as np


class StepType:
    DataCleaning = 'DataCleanStep'
    CollinearityDetection = 'MulticollinearityDetectStep'
    DriftDetection =  'DriftDetectStep'
    SpaceSearch = 'SpaceSearchStep'
    FeatureSelection = 'FeatureImportanceSelectionStep'
    PseudoLabeling  = 'PseudoLabelStep'
    DaskPseudoLabelStep  = 'DaskPseudoLabelStep'
    FeatureGeneration  = 'FeatureGenerationStep'
    PermutationImportanceSelection  = 'PermutationImportanceSelectionStep'
    ReSpaceSearch = 'ReSpaceSearch'
    Ensemble = 'EnsembleStep'
    DaskEnsembleStep = 'DaskEnsembleStep'
    FinalTrain = 'FinalTrainStep'

class StepStatus:
    Wait = 'wait'
    Process = 'process'
    Finish = 'finish'
    Skip = 'skip'
    Error = 'error'

class EarlyStoppingConfig(object):

    def __init__(self, enable, excepted_reward, max_no_improved_trials, time_limit, mode):
        self.enable = enable
        self.excepted_reward = excepted_reward
        self.max_no_improved_trials = max_no_improved_trials
        self.time_limit = time_limit
        self.mode = mode


    def to_dict(self):
        return {
            "enable": self.enable,
            "exceptedReward": self.excepted_reward,
            "maxNoImprovedTrials": self.max_no_improved_trials,
            "timeLimit": self.time_limit,
            "mode": self.mode,
        }

class EarlyStoppingStatus(object):

    def __init__(self, best_reward, best_trial_no, counter_no_improvement_trials, triggered, triggered_reason, elapsed_time):
        self.best_reward = best_reward
        self.best_trial_no = best_trial_no
        self.counter_no_improvement_trials = counter_no_improvement_trials
        self.triggered = triggered
        self.triggered_reason = triggered_reason
        self.elapsed_time = elapsed_time

    def to_dict(self):
        return {
             "bestReward": self.best_reward,
             "bestTrialNo": self.best_trial_no,
             "counterNoImprovementTrials": self.counter_no_improvement_trials,
             "triggered": self.triggered,
             "triggeredReason": self.triggered_reason,
             "elapsedTime": self.elapsed_time
        }


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
    # STATUS_NONE = -1
    # STATUS_SUCCESS = 0
    # STATUS_FAILED = 1
    # STATUS_SKIPPED = 2
    # STATUS_RUNNING = 10

    status_mapping = {
        -1: StepStatus.Wait,
        0: StepStatus.Finish,
        1:  StepStatus.Error,
        2:  StepStatus.Skip,
        10:  StepStatus.Skip,
    }
    s = step.status_
    if s not in status_mapping:
        raise Exception("Unseen status: " + str(s));
    return status_mapping[s]

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
        if 'scorer' in configuration:
            configuration['scorer'] = str(configuration['scorer'])
        return configuration

    def get_output_features(self):
        return None

    def get_extension(self, copy_ext=True):
        if get_step_status(self.step) != StepStatus.Finish:
            return {}
        else:
            extension = self.step.get_fitted_params()
            if copy_ext:
                extension = copy.deepcopy(extension)
            if isinstance(self.step.input_features_, np.ndarray):
                _inputs_list = self.step.input_features_.tolist()
            else:
                _inputs_list = self.step.input_features_

            _outputs_list = self.get_output_features()
            if _outputs_list is not None:
                increased = list(set(_outputs_list) - set(_inputs_list))
                reduced = list(set(_inputs_list) - set(_outputs_list))
            else:  # has no output
                increased = None
                reduced = None

            features_data = {
                'inputs': _inputs_list,
                'outputs': _outputs_list,
                'increased': increased,
                'reduced': reduced
            }
            extension['features'] = features_data
            return self.handle_extenion(extension)

    def handle_extenion(self, extension):
        return extension

class ABS_FeatureSelectionStepExtractor(Extractor):

    def get_output_features(self):
        selected_features = self.step.selected_features_  #
        if selected_features is None:
            return self.step.input_features_
        else:
            return selected_features

class extract_data_clean_step(ABS_FeatureSelectionStepExtractor):
    def get_output_features(self):
        return self.step.input_features_

class extract_feature_generation_step(Extractor):
    def get_output_features(self):
        output_features = self.step.transformer_.transformed_feature_names_
        if output_features is None:
            return self.step.input_features_
        else:
            return output_features

    def handle_extenion(self, extension):
        transformer = self.step.transformer_
        replaced_feature_defs_names = transformer.transformed_feature_names_
        feature_defs_names = transformer.feature_defs_names_

        mapping = None
        if replaced_feature_defs_names is not None and len(replaced_feature_defs_names) > 0 and feature_defs_names is not None and len(feature_defs_names) > 0:
            if len(replaced_feature_defs_names) == len(feature_defs_names):
                mapping = {}
                for (k, v) in zip(feature_defs_names, replaced_feature_defs_names):
                    mapping[k] = v
            else:
                pass  # replaced_feature_defs_names or feature_defs_names missing match
        else:
            pass  # replaced_feature_defs_names or feature_defs_names is empty

        def get_feature_detail(f):
            f_name = f.get_name()
            if mapping is not None:
                f_name = mapping.get(f_name)
            return {
                'name': f_name,
                'primitive': type(f.primitive).__name__,
                'parentFeatures': list(map(lambda x: x.get_name(), f.base_features)),
                'variableType': f.variable_type.type_string,
                'derivationType': type(f).__name__
            }
        feature_defs = self.step.transformer_.feature_defs_
        output_features = list(map(lambda f: get_feature_detail(f), feature_defs))
        extension = {"outputFeatures": output_features , 'features': extension['features']}
        return extension

class extract_drift_step(ABS_FeatureSelectionStepExtractor):

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
                if removed_features is not None and len(removed_features) > 0:  # ignore empty epoch
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

class abs_feature_selection_step(ABS_FeatureSelectionStepExtractor):

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
        output_extension = self.build_importances_result_(self.step.input_features_, imps, self.step.selected_features_)
        output_extension['features'] = extension['features']
        return output_extension

class extract_multi_linearity_step(ABS_FeatureSelectionStepExtractor):

    def handle_extenion(self, extension):
        feature_clusters = extension['feature_clusters']
        unselected_features = []
        for fs in feature_clusters:
            if len(fs) > 1:
                reserved = fs.copy().pop(0)
                for f_i, remove in enumerate(fs):
                    if f_i > 0:  # drop first element
                        unselected_features.append({"removed": remove, "reserved": reserved})
        output_extension = {'unselected_features': unselected_features , 'features': extension['features']}
        return output_extension

# class extract_psedudo_step(Extractor):
#     def get_configuration(self):
#         configuration = super(extract_psedudo_step, self).get_configuration()
#         del configuration['estimator_builder']
#         del configuration['estimator_builder__scorer']
#         del configuration['name']
#         return configuration
#
#     def handle_extenion(self, extension):
#         # step.estimator_builder.estimator_.classes_
#         # step.test_proba_
#         # step.pseudo_label_stat_
#         return extension

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

        output_extension = self.build_importances_result_(columns, importances_data, selected_features)
        output_extension['features'] = extension['features']
        return output_extension

class extract_space_search_step(Extractor):
    def handle_extenion(self, extension):
        extension['history'] = None
        return extension

    def get_configuration(self):
        configs = super(extract_space_search_step, self).get_configuration()
        callbacks = self.step.experiment.hyper_model.callbacks
        earlyStoppingConfig = EarlyStoppingConfig(False, None, None, None, None)
        if callbacks is not None and len(callbacks) > 0:
            for c in callbacks:
                if c.__class__.__name__ == 'EarlyStoppingCallback':
                    earlyStoppingConfig = EarlyStoppingConfig(True, c.expected_reward, c.max_no_improvement_trials , c.time_limit, c.mode)
                    break
        configs['earlyStopping'] = earlyStoppingConfig.to_dict()
        return configs


class extract_final_train_step(Extractor):

    def get_extension(self):

        if get_step_status(self.step) != StepStatus.Finish:
            return {}
        else:
            extension = super(extract_final_train_step, self).get_extension()
            extension["estimator"] = self.step.estimator_.gbm_model.__class__.__name__
            return extension

class extract_ensemble_step(Extractor):

    def get_configuration(self):
        configuration = super(extract_ensemble_step, self).get_configuration()
        return configuration

    def get_extension(self):

        if get_step_status(self.step) != StepStatus.Finish:
            return {}
        else:
            extension = super(extract_ensemble_step, self).get_extension(copy_ext=False)
            ensemble = extension['estimator']
            return {
                'weights': np.array(ensemble.weights_).tolist(),
                'scores': np.array(ensemble.scores_).tolist(),
                'features': extension['features']
            }
            return extension

class extract_psedudo_step(Extractor):
    def get_output_features(self):
        return self.step.input_features_

    def get_configuration(self):
        configuration = super(extract_psedudo_step, self).get_configuration()
        # del configuration['estimator_builder']
        # del configuration['estimator_builder__scorer']
        # del configuration['name']
        return configuration

    def handle_extenion(self, extension):
        pseudo_label_stat = self.step.pseudo_label_stat_
        classes_ = list(pseudo_label_stat.keys()) if pseudo_label_stat is not None else None

        scores = self.step.test_proba_

        if pseudo_label_stat is not None:
            for k, v in  pseudo_label_stat.items():
                if hasattr(v, 'tolist'):
                    pseudo_label_stat[k] = v.tolist()
            pseudo_label_stat = dict(pseudo_label_stat)
        else:
            pseudo_label_stat = {}

        if scores is not None and np.shape(scores)[0] > 0 and classes_ is not None and np.shape(classes_)[0] > 0:
            probability_density = self.get_proba_density_estimation(scores, classes_)
        else:
            probability_density = {}

        result_extension = \
            {
                "probabilityDensity": probability_density,
                "samples": pseudo_label_stat,
                "selectedLabel": classes_[0],
                'features': extension['features'],
            }
        return result_extension

    #@staticmethod
    # def get_proba_density_estimation(y_proba_on_test, classes):
    #     from sklearn.neighbors import KernelDensity
    #     total_class = len(classes)
    #     total_proba = np.size(y_proba_on_test, 0)
    #
    #     true_density = [[0] * 501 for _ in range(total_class)]
    #     X_plot_true_density = np.linspace(0, 1, 501)[:, np.newaxis]
    #     X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
    #
    #     probability_density = {}
    #     for i in range(total_class):
    #         aclass = classes[i]
    #         probability_density[str(aclass)] = {}
    #
    #         # calculate the true density
    #         proba_list = y_proba_on_test[:, i]
    #         probability_density[str(aclass)]['nSamples'] = len(proba_list)
    #         for proba in proba_list:
    #             true_density[i][int(proba * 500)] += 1
    #         probability_density[str(aclass)]['trueDensity'] = {}
    #         probability_density[str(aclass)]['trueDensity']['X'] = X_plot_true_density
    #         probability_density[str(aclass)]['trueDensity']['probaDensity'] = list(
    #             map(lambda x: x / total_proba, true_density[i]))
    #
    #         # calculate the gaussian/tophat/epanechnikov density estimation
    #         proba_list_2d = y_proba_on_test[:, i][:, np.newaxis]
    #         kernels = ['gaussian', 'tophat', 'epanechnikov']
    #         for kernel in kernels:
    #             kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(proba_list_2d)
    #             log_dens = kde.score_samples(X_plot)
    #             probability_density[str(aclass)][str(kernel)] = {}
    #             probability_density[str(aclass)][str(kernel)]['X'] = X_plot
    #             probability_density[str(aclass)][str(kernel)]['probaDensity'] = np.exp(log_dens)
    #     return probability_density


    @staticmethod
    def get_proba_density_estimation(scores, classes, n_partitions=1000):
        # from sklearn.neighbors import KernelDensity
        probability_density = {}
        from seaborn._statistics import KDE
        for i, class_ in enumerate(classes):
            selected_proba = np.array(scores[:, i])
            selected_proba_series = pd.Series(selected_proba).dropna()
            # selected_proba = selected_proba.reshape((selected_proba.shape[0], 1))
            estimator = KDE(bw_method='scott', bw_adjust=0.01, gridsize=200, cut=3, clip=None, cumulative=False)
            density, support = estimator(selected_proba_series, weights=None)
            probability_density[class_] = {
                'gaussian': {
                    "X": support.tolist(),
                    "probaDensity": density.tolist()
                }
            }
            # X_plot = np.linspace(0, 1, n_partitions)[:, np.newaxis]
            # # calculate the gaussian/tophat/epanechnikov density estimation
            # kernels = ['gaussian']
            # # kernels = ['gaussian', 'tophat', 'epanechnikov']
            # for kernel in kernels:
            #     kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(selected_proba)
            #     log_dens = kde.score_samples(X_plot)
            #     probability_density[class_] = {
            #         kernel: {
            #             "X": np.copy(X_plot).flatten().tolist(),
            #             "probaDensity": np.exp(log_dens).tolist()
            #         }
            #     }
        return probability_density


extractors = {
    StepType.DataCleaning: extract_data_clean_step,
    StepType.FeatureGeneration: extract_feature_generation_step,
    StepType.DriftDetection: extract_drift_step,
    StepType.FeatureSelection: extract_feature_selection_step,
    StepType.CollinearityDetection: extract_multi_linearity_step,
    StepType.PseudoLabeling: extract_psedudo_step,
    StepType.DaskPseudoLabelStep: extract_psedudo_step,
    StepType.PermutationImportanceSelection: extract_permutation_importance_step,
    StepType.SpaceSearch: extract_space_search_step,
    StepType.FinalTrain: extract_final_train_step,
    StepType.Ensemble: extract_ensemble_step,
    StepType.DaskEnsembleStep: extract_ensemble_step
}

def extract_step(index, step):
    stepType = step.__class__.__name__

    extractor_cls = extractors.get(stepType)
    if extractor_cls is None:
        raise Exception(f"Unseen Step class {stepType} ")
    extractor = extractor_cls(step)
    configuration = extractor.get_configuration()
    extension = extractor.get_extension()

    d = \
        StepData(index=index,
                 name=step.name,
                 type=step.__class__.__name__,
                 status=get_step_status(step),
                 configuration=configuration,
                 extension=extension,
                 start_datetime=step.start_time,
                 end_datetime=step.done_time)
    return d.to_dict()


def extract_experiment(compete_experiment):
    step_dict_list = []
    for i, step in enumerate(compete_experiment.steps):
        step_dict_list.append(extract_step(i, step))
    return {"steps": step_dict_list}
