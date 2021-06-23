export const Steps = {
    DataCleaning: {
        name: 'Data cleaning',
        type: 'DataCleanStep',
        configTip: {}
    },
    FeatureGeneration: {
        name: 'Feature generation',
        type: 'FeatureGenerationStep',
        configTip: {
            strategy:  "Strategy to select features",
            threshold:  "Confidence threshold of feature_importance. Only valid when *feature_selection_strategy* is 'threshold'.",
            quantile:  "Confidence quantile of feature_importance. Only valid when *feature_selection_strategy* is 'quantile'.",
            number: "Expected feature number to keep. Only valid when *feature_selection_strategy* is 'number'.",
        }
    },
    CollinearityDetection: {
        name: 'Collinearity detection',
        type: 'MulticollinearityDetectStep',
        configTip: {}
    },
    DriftDetection: {
        name: 'Drift detection',
        type: 'DriftDetectStep',
        configTip: {
            // remove_shift_variable: "",
            // variable_shift_threshold: "",
            // threshold: "",
            // remove_size:"",
            // min_features:"",
            // num_folds:"",
        }
    },
    SpaceSearch: {
        name: 'Pipeline optimization',
        type: 'SpaceSearchStep',
        key: 'space_searching',
        configTip: {
            cv: "If True, use cross-validation instead of evaluation set reward to guide the search process",
            num_folds: "Number of cross-validated folds, only valid when cv is true"
        }
    },
    FeatureSelection: {
        name: 'Feature selection',
        type: 'FeatureImportanceSelectionStep',
        configTip: {
            feature_reselection: "Whether to enable two stage feature selection with permutation importance.",
            estimator_size: "The number of estimator to evaluate feature importance. Only valid when *feature_reselection* is True.",
            threshold: "Confidence threshold of the mean permutation importance. Only valid when *feature_reselection_strategy* is 'threshold'.",
        }
        // drift_detection
    },
    PsudoLabeling: {
        name: 'Psudo labeling',
        type: 'PseudoLabelStep',
        configTip: {
            proba_threshold: "Confidence threshold of pseudo-label samples. Only valid when *pseudo_labeling_strategy* is 'threshold'.",
            resplit: "Whether to re-split the training set and evaluation set after adding pseudo-labeled data. If False, the pseudo-labeled data is only appended to the training set. Only valid when *pseudo_labeling* is True.",
            strategy: "Strategy to sample pseudo labeling data(*threshold*, *number* or *quantile*)."
        }
    },
    PermutationImportanceSelection: {
        name: 'Permutation importance selection',
        type: 'PermutationImportanceSelectionStep',
        configTip: {
            estimator_size: "The number of estimator to evaluate feature importance. Only valid when *feature_reselection* is True.",
            strategy: "Strategy to reselect features(*threshold*, *number* or *quantile*).",
            threshold: "Confidence threshold of the mean permutation importance. Only valid when *feature_reselection_strategy* is 'threshold'.",
            quantile: "Confidence quantile of feature_importance. Only valid when *feature_reselection_strategy* is 'quantile'.",
            number: "Expected feature number to keep. Only valid when *feature_reselection_strategy* is 'number'."
        }
    },
    TwoStageSpaceSearch: {
        name: 'Two-stage pipeline optimization',
        type: 'SpaceSearchStep',
        key: 'two_stage_searching'
    },
    FinalTrain: {
        name: 'Final train',
        type: 'FinalTrainStep',
        configTip: {}
    },
    Ensemble: {
        name: 'Ensemble',
        type: 'EnsembleStep',
        configTip: {
            scorer: "Scorer to used for feature importance evaluation and ensemble."
        }
    }
};



export function getStepName(stepType) {
    const keys = Object.keys(Steps);
    for (let i = 0; i < keys.length; i++) {
        const k = keys[i];
        const v = Steps[k];
        if(v.type === stepType){
            return v.name;
        }
    }
    return null;
}

export const StepStatus = {
  Wait: 'wait',
  Process: 'process',
  Finish: 'finish',
  Error: 'error'
};

export const ActionType = {
    EarlyStopped: 'earlyStopped',
    StepFinished: 'stepFinished',
    StepBegin: 'stepBegin',
    StepError: 'stepError',
    TrialFinished: 'trialFinished',
    ProbaDensityLabelChange: 'probaDensityLabelChange',
    ExperimentData: 'experimentData'
};



export const MAX_FEATURES_OF_IMPORTANCES = 10;

export const COMPONENT_SIZE = 'small';
export const TABLE_ITEM_SIZE = '5';

