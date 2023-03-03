import json
import abc
import copy
from collections import OrderedDict, namedtuple
from typing import List, Set, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

from hypernets.model import Estimator
from hypernets.utils import logging, get_tree_importances


logger = logging.get_logger(__name__)


class StepType:
    DataAdapting = 'DataAdaptionStep'
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


class BaseMeta(metaclass=abc.ABCMeta):

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.to_dict())

    @staticmethod
    def _to_dict_list(meta_list: List):
        if meta_list is not None:
            return [meta_data.to_dict() for meta_data in meta_list]
        else:
            return None


class EarlyStoppingConfigMeta(BaseMeta):

    def __init__(self, enable, excepted_reward,max_no_improved_trials, time_limit, mode):
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


class EarlyStoppingStatusMeta(BaseMeta):

    def __init__(self, best_reward, best_trial_no, counter_no_improvement_trials,
                 triggered, triggered_reason, elapsed_time):
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


class DatasetMeta(BaseMeta):

    TYPE_TRAIN = 'Train'
    TYPE_TEST = 'Test'
    TYPE_EVAL = 'Eval'

    def __init__(self, kind, task, shape, memory):
        self.kind = kind
        self.task = task
        self.shape = shape
        self.memory = memory


class StepMeta(BaseMeta):

    STATUS_WAIT = 'wait'
    STATUS_PROCESS = 'process'
    STATUS_FINISH = 'finish'
    STATUS_SKIP = 'skip'
    STATUS_ERROR = 'error'

    def __init__(self, index, name, type, status, configuration, extension, start_datetime, end_datetime):
        self.index = index
        self.name = name
        self.type = type
        self.status = status
        self.configuration = configuration
        self.extension = extension
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime


class ConfusionMatrixMeta(BaseMeta):

    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels


class ExperimentMeta(BaseMeta):

    def __init__(self, task, datasets: List[DatasetMeta], steps: List[StepMeta], resource_usage=None,
                 evaluation_metrics=None, classification_report=None,
                 confusion_matrix: ConfusionMatrixMeta = None, prediction_elapsed=None):

        self.task = task
        self.datasets = datasets
        self.steps = steps
        self.resource_usage = resource_usage

        self.evaluation_metrics = evaluation_metrics
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report
        self.prediction_elapsed = prediction_elapsed  # prediction_stats

    def to_dict(self):
        dict_data = self.__dict__
        dict_data['datasets'] = self._to_dict_list(self.datasets)
        dict_data['steps'] = self._to_dict_list(self.steps)
        return dict_data


class Extractor(metaclass=abc.ABCMeta):

    def __init__(self, step):
        self.step = step

    def get_configuration(self):
        configuration = copy.deepcopy(self.step.get_params())
        if 'scorer' in configuration:
            configuration['scorer'] = str(configuration['scorer'])
        return configuration

    @abc.abstractmethod
    def selected_features(self):
        raise NotImplemented

    def get_status(self):
        status_mapping = {
            -1: StepMeta.STATUS_WAIT, 0: StepMeta.STATUS_FINISH, 1: StepMeta.STATUS_ERROR,
            2: StepMeta.STATUS_SKIP, 10: StepMeta.STATUS_SKIP,
        }
        s = self.step.status_
        if s not in status_mapping:
            raise ValueError("Unseen status: " + str(s))
        return status_mapping[s]

    def get_step_cls_name(self):
        class_name = self.step.__class__.__name__
        return class_name

    def _get_features(self):
        if isinstance(self.step.input_features_, np.ndarray):
            inputs_list = self.step.input_features_.tolist()
        else:
            inputs_list = self.step.input_features_

        outputs_list = self.selected_features()

        if outputs_list is not None:
            increased = list(set(outputs_list) - set(inputs_list))
            reduced = list(set(inputs_list) - set(outputs_list))
        else:  # has no output
            increased = None
            reduced = None

        features_data = {
            'inputs': inputs_list,
            'outputs': outputs_list,
            'increased': increased,
            'reduced': reduced
        }
        return features_data

    def get_extension(self):
        if self.get_status() != StepMeta.STATUS_FINISH:
            return {}
        else:
            ret_extension = self._get_extension()
            ret_extension['features'] = self._get_features()
            return ret_extension

    @abc.abstractmethod
    def _get_extension(self):
        raise NotImplemented


class ABSFeatureSelectionStepExtractor(Extractor, metaclass=abc.ABCMeta):

    # def _get_extension(self):
    #     selected_features = self.step.selected_features_  #
    #     if selected_features is None or len(selected_features) == 0:
    #         raise ValueError(f"'selected_features_' is empty for step {self.step}")

    def selected_features(self):
        return self.step.selected_features

    def unselected_features(self):
        return self.step.unselected_features


class DataAdaptionStepExtractor(ABSFeatureSelectionStepExtractor):

    def _get_extension(self):
        return {}


class DataCleanStepExtractor(ABSFeatureSelectionStepExtractor):

    def _get_extension(self):
        return {'unselected_reason': self.step.get_fitted_params()['unselected_reason']}


class FeatureGenerationStepExtractor(Extractor):
    def selected_features(self):
        output_features = self.step.transformer_.transformed_feature_names_
        if output_features is None:
            return self.step.input_features_
        else:
            return output_features

    def _get_extension(self):
        """
        Parameters
        ----------

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

        def get_variable_type(f):
            if hasattr(f, 'variable_type'):
                return f.variable_type.type_string
            elif hasattr(f, 'column_schema'):
                column_schema = f.column_schema
                if column_schema.is_boolean:
                    return 'boolean'
                if column_schema.is_categorical:
                    return 'categorical'
                if column_schema.is_datetime:
                    return 'datetime'
                if column_schema.is_numeric:
                    return 'numeric'
                return 'unknown'
            return 'unknown_ft_version'

        def get_feature_detail(f):
            f_name = f.get_name()
            if mapping is not None:
                f_name = mapping.get(f_name)
            return {
                'name': f_name,
                'primitive': type(f.primitive).__name__,
                'parentFeatures': list(map(lambda x: x.get_name(), f.base_features)),
                'variableType': get_variable_type(f),
                'derivationType': type(f).__name__
            }
        feature_defs = self.step.transformer_.feature_defs_
        output_features = list(map(lambda f: get_feature_detail(f), feature_defs))
        extension = {"outputFeatures": output_features}
        return extension


class DriftStepExtractor(ABSFeatureSelectionStepExtractor):

    def _get_extension(self):
        extension = self.step.get_fitted_params()
        config = super(DriftStepExtractor, self).get_configuration()
        extension['drifted_features_auc'] = []
        over_variable_threshold_features = []
        if 'scores' in extension and extension['scores'] is not None:
            scores = extension['scores']
            variable_shift_threshold = config['variable_shift_threshold']
            if config['remove_shift_variable']:
                for col, score in scores.items():
                    if col not in self.selected_features():
                        if score > variable_shift_threshold:
                            over_variable_threshold_features.append((col, score))  # removed
                        else:
                            pass
                            # logger.warning(f"Score of feature '{col}' is "
                            #                f"{score} < threshold = {variable_shift_threshold} but not removed.")
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


class ABSFeatureImportancesSelectionStepExtractor(ABSFeatureSelectionStepExtractor, metaclass=abc.ABCMeta):

    def _build_importances(self, columns, importances_data, selected_features):
        features = []
        for col, imp in zip(columns, importances_data):
            features.append({
                'name': col,
                'importance': imp,
                'dropped': col not in selected_features
            })

        return sorted(features, key=lambda v: v['importance'], reverse=True)


class FeatureSelectionStepExtractor(ABSFeatureImportancesSelectionStepExtractor):
    def _get_extension(self):
        """
        Parameters
        ----------
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
        imps = self.step.get_fitted_params()['importances']
        # extension['importances'] = imps.tolist() if imps is not None else []
        output_extension = {
            "importances":  self._build_importances(self.step.input_features_, imps, self.selected_features())
        }
        return output_extension


class MultiLinearityStepExtractor(ABSFeatureSelectionStepExtractor):

    def _get_extension(self):
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
        feature_clusters = self.step.get_fitted_params()['feature_clusters']
        unselected_features = OrderedDict()
        for fc in feature_clusters:
            if len(fc) > 1:
                reserved = fc[0]
                for f_i, remove in enumerate(fc):
                    if f_i > 0:  # drop first element
                        unselected_features[remove] = {'reserved': reserved}
        return {'unselected_features': unselected_features}


class PermutationImportanceStepExtractor(ABSFeatureImportancesSelectionStepExtractor):
    def get_configuration(self):
        configuration = super(PermutationImportanceStepExtractor, self).get_configuration()
        configuration['scorer'] = str(configuration['scorer'])
        return configuration

    def _get_extension(self):

        selected_features = self.selected_features()
        importances = self.step.importances_
        columns = importances.columns if importances.columns is not None else []
        importances_data = importances.importances_mean.tolist() if importances.importances_mean is not None else []

        output_extension = {
            "importances": self._build_importances(columns, importances_data, selected_features)
        }
        return output_extension


class SpaceSearchStepExtractor(Extractor):
    def _get_extension(self):
        ret_ext = self.step.get_fitted_params()
        ret_ext_copy = copy.deepcopy(ret_ext)
        ret_ext_copy['history'] = None
        return ret_ext_copy

    def selected_features(self):
        return ['y']

    def get_configuration(self):
        configs = super(SpaceSearchStepExtractor, self).get_configuration()
        callbacks = self.step.experiment.hyper_model.callbacks
        earlyStoppingConfig = EarlyStoppingConfigMeta(False, None, None, None, None)
        if callbacks is not None and len(callbacks) > 0:
            for c in callbacks:
                if c.__class__.__name__ == 'EarlyStoppingCallback':
                    earlyStoppingConfig = EarlyStoppingConfigMeta(True, c.expected_reward, c.max_no_improvement_trials , c.time_limit, c.mode)
                    break
        configs['earlyStopping'] = earlyStoppingConfig.to_dict()
        return configs


class FinalTrainStepExtractor(Extractor):

    def get_extension(self):

        if self.get_status() != StepMeta.STATUS_FINISH:
            return {}
        else:
            extension = super(FinalTrainStepExtractor, self).get_extension()
            extension["estimator"] = self.step.estimator_.gbm_model.__class__.__name__
            return extension

    def selected_features(self):
        return ['y']

    def _get_extension(self):
        return {}


class EnsembleStepExtractor(Extractor):

    def selected_features(self):
        return ['y']

    def get_configuration(self):
        configuration = super(EnsembleStepExtractor, self).get_configuration()
        return configuration

    def _get_extension(self):
        ensemble = self.step.estimator_

        def get_models(estimator_: Estimator):
            if estimator_.cv_ is not None:
                if estimator_.cv_:
                    assert estimator_.cv_models_, "cv is enable but cv models is None"
                    return estimator_.cv_models_
                else:
                    assert estimator_.model, "model can not be None"
                    return [estimator_.model]
            else:
                raise RuntimeError("there is no models detected")

        estimators = []
        for i, estimator in enumerate(ensemble.estimators):
            if estimator is not None:
                _e_mate = {
                    'index': i,
                    'weight': ensemble.weights_[i],
                    'lift': ensemble.scores_[i],
                    'models': [get_tree_importances(m) for m in get_models(estimator)]
                }
                estimators.append(_e_mate)
        return {'estimators': estimators}


class PseudoStepExtractor(Extractor):
    def selected_features(self):
        return self.step.input_features_

    def get_configuration(self):
        configuration = super(PseudoStepExtractor, self).get_configuration()
        # del configuration['estimator_builder']
        # del configuration['estimator_builder__scorer']
        # del configuration['name']
        return configuration

    def _get_extension(self):

        def np2py(o):
            if hasattr(o, 'tolist'):
                return o.tolist()
            else:
                return o

        pseudo_label_stat = self.step.pseudo_label_stat_
        classes_ = list(map(lambda _: np2py(_), pseudo_label_stat.keys())) if pseudo_label_stat is not None else None

        scores = self.step.test_proba_

        if pseudo_label_stat is not None:
            pseudo_label_stat_ = {}
            for k, v in pseudo_label_stat.items():
                pseudo_label_stat_[np2py(k)] = np2py(v)
            pseudo_label_stat = dict(pseudo_label_stat_)
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
                "selectedLabel": np2py(classes_[0]),
            }
        return result_extension

    @staticmethod
    def get_proba_density_estimation(scores, classes, n_partitions=1000):
        probability_density = {}

        def calc(proba):
            cut = 3
            gridsize = 200

            kde = stats.gaussian_kde(proba, bw_method='scott')
            kde.set_bandwidth(0.01 * kde.factor)
            bw = np.sqrt(kde.covariance.squeeze())
            gridmin = max(proba.min() - bw * cut, -np.inf)
            gridmax = min(proba.max() + bw * cut, +np.inf)
            support = np.linspace(gridmin, gridmax, gridsize)
            return kde(support), support

        for i, class_ in enumerate(classes):
            selected_proba = np.array(scores[:, i])
            proba = selected_proba[~np.isnan(selected_proba)]
            # selected_proba = selected_proba.reshape((selected_proba.shape[0], 1))
            density, support = calc(proba)
            probability_density[class_] = {
                'gaussian': {
                    "X": support.tolist(),
                    "probaDensity": density.tolist()
                }
            }
        return probability_density


class ExperimentExtractor:

    def __init__(self, exp, resource_usage=None):
        self.exp = exp
        self.resource_usage = resource_usage

    extractors = {
        StepType.DataAdapting: DataAdaptionStepExtractor,
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

    @staticmethod
    def extract_step(index, step):
        step_type = step.__class__.__name__

        extractor_cls = ExperimentExtractor.extractors.get(step_type)
        if extractor_cls is None:
            raise Exception(f"Unseen Step class {step_type} ")
        extractor = extractor_cls(step)
        configuration = extractor.get_configuration()
        extension = extractor.get_extension()

        d = \
            StepMeta(index=index,
                     name=step.name,
                     type=step.__class__.__name__,
                     status=extractor.get_status(),
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

    def _append_dataset_meta(self, meta_list, df, type_name, task):
        if df is not None:
            # self._get_dataset_meta(exp.X_train, 'Train', exp.task)
            meta_list.append(self._get_dataset_meta(df, type_name, task))

    def extract(self):
        exp = self.exp
        # datasets
        datasets_meta: List[DatasetMeta] = []
        self._append_dataset_meta(datasets_meta, exp.X_train, DatasetMeta.TYPE_TRAIN,  exp.task)
        self._append_dataset_meta(datasets_meta, exp.X_test, DatasetMeta.TYPE_TEST,  exp.task)
        self._append_dataset_meta(datasets_meta, exp.X_eval, DatasetMeta.TYPE_EVAL,  exp.task)

        # steps
        steps_meta = [self.extract_step(i, step) for i, step in enumerate(exp.steps)]

        # prediction stats
        if exp.evaluation_ is not None:
            evaluation = exp.evaluation_
        else:
            evaluation = {}

        # FIXME: exp.hyper_model_.task
        return ExperimentMeta(task=exp.hyper_model.task, datasets=datasets_meta,
                              steps=steps_meta, resource_usage=self.resource_usage,
                              **evaluation)
