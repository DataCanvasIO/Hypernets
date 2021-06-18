export const StepsKey = {
    DataCleaning: {
        name: 'Data cleaning',
        type: 'DataCleanStep'
    },
    FeatureGeneration: {
        name: 'Feature generation',
        type: 'FeatureGenerationStep'
    },
    CollinearityDetection: {
        name: 'Collinearity detection',
        type: 'MulticollinearityDetectStep'
    },
    DriftDetection: {
        name: 'Drift detection',
        type: 'DriftDetectStep'
    },
    SpaceSearch: {
        name: 'Pipeline optimization',
        type: 'SpaceSearchStep',
        key: 'space_searching'
    },
    FeatureSelection: {
        name: 'Feature selection',
        type: 'FeatureImportanceSelectionStep'
        // drift_detection
    },
    PsudoLabeling: {
        name: 'Psudo labeling',
        type: 'PseudoLabelStep'
    },
    PermutationImportanceSelection: {
        name: 'Permutation importance selection',
        type: 'PermutationImportanceSelectionStep'
    },
    TwoStageSpaceSearch: {
        name: 'Two-stage pipeline optimization',
        type: 'SpaceSearchStep',
        key: 'two_stage_searching'
    },
    FinalTrain: {
        name: 'Final train',
        type: 'FinalTrainStep'
    },
    Ensemble: {
        name: 'Ensemble',
        type: 'EnsembleStep'
    }
};



export function getStepName(stepType) {
    const keys = Object.keys(StepsKey);
    for (let i = 0; i < keys.length; i++) {
        const k = keys[i];
        const v = StepsKey[k];
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
