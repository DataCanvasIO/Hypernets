import json
import copy
from collections import OrderedDict, namedtuple
from typing import List, Set, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from hypernets.utils import logging


logger = logging.get_logger(__name__)


class StepType:
    DataCleaning = 'DataCleanStep'
    CollinearityDetection = 'MulticollinearityDetectStep'
    DriftDetection = 'DriftDetectStep'
    SpaceSearch = 'SpaceSearchStep'
    FeatureSelection = 'FeatureImportanceSelectionStep'
    PseudoLabeling = 'PseudoLabelStep'
    DaskPseudoLabelStep = 'DaskPseudoLabelStep'
    FeatureGeneration = 'FeatureGenerationStep'
    PermutationImportanceSelection = 'PermutationImportanceSelectionStep'
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


class DatasetMeta(namedtuple('DatasetMeta', ['kind', 'task', 'shape', 'memory'])):
    pass


class StepMeta(namedtuple('StepMeta', ['index', 'name',  'type', 'status',
                                       'configuration', 'extension', 'start_datetime', 'end_datetime'])):

    def to_dict(self):
        return self._asdict()

    def to_json(self):
        return json.dumps(self.to_dict())


class ExperimentMeta:
    def __init__(self, task, datasets_meta, steps_meta, evaluation_metric=None, confusion_matrix=None,
                 resource_usage=None, prediction_stats=None):
        self.task = task
        self.datasets_meta = datasets_meta
        self.steps_data = steps_meta
        self.evaluation_metric = evaluation_metric
        self.confusion_matrix = confusion_matrix
        self.resource_usage = resource_usage
        self.prediction_stats = prediction_stats


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


import pickle
import time
import sys
import numpy as np

from IPython.display import display

from hypernets.core.callbacks import Callback
from hypernets.core.callbacks import EarlyStoppingCallback
from hypernets.experiment import ExperimentCallback
from hypernets.utils import fs

try:
    import hn_widget
except Exception as e:
    sys.stderr.write("Please install the hypernets widget with command: pip install hypernets-jupyter-widget ")

from hn_widget.experiment_util import EarlyStoppingStatus
from hn_widget.experiment_util import get_step_index
from hn_widget.experiment_util import StepStatus

MAX_IMPORTANCE_NUM = 10  # TOP N important features


def get_tree_importances(tree_model):
    # TODO: refactor
    def _is_light_gbm_model(m):
        try:
            from lightgbm.sklearn import LGBMModel
            return isinstance(m, LGBMModel)
        except Exception as e:
            return False

    def _is_xgboost_model(m):
        try:
            from xgboost.sklearn import XGBModel
            return isinstance(m, XGBModel)
        except Exception as e:
            return False

    def _is_catboost_model(m):
        try:
            from catboost.core import CatBoost
            return isinstance(m, CatBoost)
        except Exception as e:
            return False

    def _is_decision_tree(m):
        try:
            from sklearn.tree import BaseDecisionTree
            return isinstance(m, BaseDecisionTree)
        except Exception as e:
            return False

    def get_imp(n_features):
        try:
            return tree_model.feature_importances_
        except Exception as e:
            return [0 for i in range(n_features)]

    if _is_xgboost_model(tree_model):
        importances_pairs = list(zip(tree_model._Booster.feature_names, get_imp(len(tree_model._Booster.feature_names))))
    elif _is_light_gbm_model(tree_model):
        if hasattr(tree_model, 'feature_name_'):
            names = tree_model.feature_name_
        else:
            names = [f'col_{i}' for i in range(tree_model.feature_importances_.shape[0])]
        importances_pairs = list(zip(names, get_imp(len(names))))
    elif _is_catboost_model(tree_model):
        importances_pairs = list(zip(tree_model.feature_names_, get_imp(len(tree_model.feature_names_))))
    elif _is_decision_tree(tree_model):
        importances_pairs = [(f'col_{i}', tree_model.feature_importances_[i]) for i in range(tree_model.feature_importances_.shape[0])]
    else:
        importances_pairs = []

    importances = {}
    numpy_num_types = [np.int, np.int32,np.int64, np.float, np.float32, np.float64]

    def is_numpy_num_type(v):
        for t in numpy_num_types:
            if isinstance(v, t) is True:
                return True
            else:
                continue
        return False

    for name, imp in importances_pairs:
        if is_numpy_num_type(imp):
            imp_value = imp.tolist()  # int64, float32, float64 has tolist
        elif isinstance(imp, float) or isinstance(imp, int):
            imp_value = imp
        else:
            imp_value = float(imp)  # force convert to float
        importances[name] = imp_value

    return importances


def sort_imp(imp_dict, sort_imp_dict):
    sort_imps = []
    for k in sort_imp_dict:
        sort_imps.append({
            'name': k,
            'imp': sort_imp_dict[k]
        })

    top_features = list(map(lambda x: x['name'], sorted(sort_imps, key=lambda v: v['imp'], reverse=True)[: MAX_IMPORTANCE_NUM]))

    imps = []
    for f in top_features:
        imps.append({
            'name': f,
            'imp': imp_dict[f]
        })
    return imps


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
            return self.handle_extension(extension)

    def handle_extension(self, extension):
        return extension


class ABSFeatureSelectionStepExtractor(Extractor):

    def get_output_features(self):
        selected_features = self.step.selected_features_  #
        if selected_features is None:
            return self.step.input_features_
        else:
            return selected_features


class DataCleanStepExtractor(ABSFeatureSelectionStepExtractor):
    pass


class FeatureGenerationStepExtractor(Extractor):
    def get_output_features(self):
        output_features = self.step.transformer_.transformed_feature_names_
        if output_features is None:
            return self.step.input_features_
        else:
            return output_features

    def handle_extension(self, extension):
        """

        Parameters
        ----------
        extension

        Returns
        -------
        {
            "outputFeatures": [
                {
                 'name': f_name,
                 'primitive': type(f.primitive).__name__,
                 'parentFeatures': list(map(lambda x: x.get_name(), f.base_features)),
                 'variableType': f.variable_type.type_string,
                 'derivationType': type(f).__name__
                }
            ]
        }
        """
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


class DriftStepExtractor(ABSFeatureSelectionStepExtractor):

    def handle_extension(self, extension):
        config = super(DriftStepExtractor, self).get_configuration()
        extension['drifted_features_auc'] = []
        over_variable_threshold_features = []
        if 'scores' in extension and extension['scores'] is not None:
            scores = extension['scores']
            variable_shift_threshold = config['variable_shift_threshold']
            if config['remove_shift_variable']:
                for col, score in scores.items():
                    if col not in self.step.selected_features_:
                        if score > variable_shift_threshold:
                            over_variable_threshold_features.append((col, score))  # removed
                        else:
                            logger.warning(f"Score of feature '{col}' is "
                                           f"{score} < threshold = {variable_shift_threshold} but not removed.")
                over_variable_threshold_features = sorted(over_variable_threshold_features, key=lambda item: item[1], reverse=True)
                extension['drifted_features_auc'] = over_variable_threshold_features

        def get_importance(col, feature_names, feature_importances):
            for i, c in enumerate(feature_names):
                if c == col:
                    return feature_importances[i]
            return 0
        removed_features_in_epochs = []
        history = extension['history']
        if history is not None and len(history) > 0:
            for i, history in enumerate(history):
                feature_names = history['feature_names']
                feature_importances = history['feature_importances']
                removed_features = [] if 'removed_features' not in history else history['removed_features']
                if isinstance(feature_importances, np.ndarray):
                    feature_importances = feature_importances.tolist()

                if removed_features is not None and len(removed_features) > 0:  # ignore empty epoch
                    removed_features_importances = [(f, get_importance(f, feature_names, feature_importances)) for f in removed_features]
                    removed_features_importances = sorted(removed_features_importances, key=lambda item: item[1], reverse=True)
                    d = {
                        "epoch": i,
                        "elapsed": history['elapsed'],
                        "removed_features": removed_features_importances
                    }
                    removed_features_in_epochs.append(d)
        del extension['scores']
        del extension['history']
        return {
            'unselected_features': {
                'over_variable_threshold': over_variable_threshold_features,
                'over_threshold': removed_features_in_epochs,
            }
        }


class ABSFeatureImportancesSelectionStepExtractor(ABSFeatureSelectionStepExtractor):

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
        return extension  # TODO: ret not dict


class FeatureSelectionStepExtractor(ABSFeatureImportancesSelectionStepExtractor):
    def handle_extension(self, extension):
        """
        Parameters
        ----------
        extension

        Returns
        -------
        {
            'importances': [
                {
                    'name': 'name',
                    'importance': '0.2',
                    'dropped': True,
                }
            ]
        }
        """
        imps = extension['importances']
        extension['importances'] = imps.tolist() if imps is not None else []
        output_extension = self.build_importances_result_(self.step.input_features_, imps, self.step.selected_features_)
        output_extension['features'] = extension['features']
        return output_extension


class MultiLinearityStepExtractor(ABSFeatureSelectionStepExtractor):

    def handle_extension(self, extension):
        """
        Parameters
        ----------
        extension

        Returns
        -------
        {
            'unselected_features': {
                'name': {
                    'reserved': 'name_labeled'
                }
            },
            'features': extension['features']
        }
        """
        feature_clusters = extension['feature_clusters']
        unselected_features = OrderedDict()
        for fc in feature_clusters:
            if len(fc) > 1:
                reserved = fc[0]
                for f_i, remove in enumerate(fc):
                    if f_i > 0:  # drop first element
                        unselected_features[remove] = {'reserved': reserved}
        output_extension = {'unselected_features': unselected_features, 'features': extension['features']}
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

class PermutationImportanceStepExtractor(ABSFeatureImportancesSelectionStepExtractor):
    def get_configuration(self):
        configuration = super(PermutationImportanceStepExtractor, self).get_configuration()
        configuration['scorer'] = str(configuration['scorer'])
        return configuration

    def handle_extension(self, extension):
        selected_features = self.step.selected_features_ if self.step.selected_features_  is not None else []
        importances = self.step.importances_
        columns = importances.columns if importances.columns is not None else []
        importances_data = importances.importances_mean.tolist() if importances.importances_mean is not None else []

        output_extension = self.build_importances_result_(columns, importances_data, selected_features)
        output_extension['features'] = extension['features']
        return output_extension


class SpaceSearchStepExtractor(Extractor):
    def handle_extension(self, extension):
        extension['history'] = None
        return extension

    def get_configuration(self):
        configs = super(SpaceSearchStepExtractor, self).get_configuration()
        callbacks = self.step.experiment.hyper_model.callbacks
        earlyStoppingConfig = EarlyStoppingConfig(False, None, None, None, None)
        if callbacks is not None and len(callbacks) > 0:
            for c in callbacks:
                if c.__class__.__name__ == 'EarlyStoppingCallback':
                    earlyStoppingConfig = EarlyStoppingConfig(True, c.expected_reward, c.max_no_improvement_trials , c.time_limit, c.mode)
                    break
        configs['earlyStopping'] = earlyStoppingConfig.to_dict()
        return configs


class FinalTrainStepExtractor(Extractor):

    def get_extension(self):

        if get_step_status(self.step) != StepStatus.Finish:
            return {}
        else:
            extension = super(FinalTrainStepExtractor, self).get_extension()
            extension["estimator"] = self.step.estimator_.gbm_model.__class__.__name__
            return extension


class EnsembleStepExtractor(Extractor):

    def get_configuration(self):
        configuration = super(EnsembleStepExtractor, self).get_configuration()
        return configuration

    def get_extension(self, copy_ext=False):
        # TODO: adapt for notebook
        if get_step_status(self.step) != StepStatus.Finish:
            return {}
        else:
            ensemble = self.step.estimator_

            estimators = []
            for i, estimator in enumerate(ensemble.estimators):
                if estimator is not None:
                    _e_mate = {
                        'index': i,
                        'weight': ensemble.weights_[i],
                        'lift': ensemble.scores_[i],
                        'models': [get_tree_importances(m) for m in estimator.cv_models_]  # FIXME: no cv
                    }
                    estimators.append(_e_mate)
            return {'estimators': estimators}


class PseudoStepExtractor(Extractor):
    def get_output_features(self):
        return self.step.input_features_

    def get_configuration(self):
        configuration = super(PseudoStepExtractor, self).get_configuration()
        # del configuration['estimator_builder']
        # del configuration['estimator_builder__scorer']
        # del configuration['name']
        return configuration

    def handle_extension(self, extension):
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
            selected_proba_series = pd.Series(selected_proba).dropna()  # todo use numpy instead to remove pandas
            # selected_proba = selected_proba.reshape((selected_proba.shape[0], 1))
            estimator = KDE(bw_method='scott', bw_adjust=0.01, gridsize=200, cut=3, clip=None, cumulative=False)
            density, support = estimator(selected_proba_series, weights=None)
            probability_density[class_] = {
                'gaussian': {
                    "X": support.tolist(),
                    "probaDensity": density.tolist()
                }
            }
        return probability_density


extractors = {
    StepType.DataCleaning: DataCleanStepExtractor,
    StepType.FeatureGeneration: FeatureGenerationStepExtractor,
    StepType.DriftDetection: DriftStepExtractor,
    StepType.FeatureSelection: FeatureSelectionStepExtractor,
    StepType.CollinearityDetection: MultiLinearityStepExtractor,
    StepType.PseudoLabeling: PseudoStepExtractor,
    StepType.DaskPseudoLabelStep: PseudoStepExtractor,
    StepType.PermutationImportanceSelection: PermutationImportanceStepExtractor,
    StepType.SpaceSearch: SpaceSearchStepExtractor,
    StepType.FinalTrain: FinalTrainStepExtractor,
    StepType.Ensemble: EnsembleStepExtractor,
    StepType.DaskEnsembleStep: EnsembleStepExtractor
}


class ExperimentExtractor:

    def __init__(self, exp, evaluation_result=None, confusion_matrix_result=None,
                 resource_usage=None):
        self.exp = exp
        self.evaluation_result = evaluation_result
        self.confusion_matrix_result = confusion_matrix_result
        self.resource_usage = resource_usage

    @staticmethod
    def _extract_step(index, step):
        step_type = step.__class__.__name__

        extractor_cls = extractors.get(step_type)
        if extractor_cls is None:
            raise Exception(f"Unseen Step class {step_type} ")
        extractor = extractor_cls(step)
        configuration = extractor.get_configuration()
        extension = extractor.get_extension()

        d = \
            StepMeta(index=index,
                     name=step.name,
                     type=step.__class__.__name__,
                     status=get_step_status(step),
                     configuration=configuration,
                     extension=extension,
                     start_datetime=step.start_time,
                     end_datetime=step.done_time)
        return d

    @staticmethod
    def _get_dataset_meta(df: pd.DataFrame, kind, task):
        if df is not None:
            return DatasetMeta(kind, task, df.shape, np.sum(df.memory_usage().values).tolist())
        else:
            return None

    def extract(self):
        exp = self.exp
        # FIXME: No eval or test
        datasets_meta: List[DatasetMeta] = [self._get_dataset_meta(exp.X_train, 'Train', exp.task),
                                            self._get_dataset_meta(exp.X_eval, 'Eval', exp.task),
                                            self._get_dataset_meta(exp.X_test, 'Test', exp.task)]

        steps_meta = [self._extract_step(i, step) for i, step in enumerate(exp.steps)]

        return ExperimentMeta(task=exp.task, datasets_meta=datasets_meta, steps_meta=steps_meta,
                              evaluation_metric=self.evaluation_result, confusion_matrix=self.confusion_matrix_result,
                              resource_usage=self.resource_usage,)
