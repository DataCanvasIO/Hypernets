export const StepsKey = {
    DataCleaning: {
        name: 'Data cleaning',
        kind: 'DataCleanStep'
    },
    CollinearityDetection: {
        name: 'Collinearity detection',
        kind: 'DriftDetectStep'},
    DriftDetection: {
        name: 'Drift detection',
        kind: 'drift_detection'
    },
    PipelineOptimization: {
        name: 'Pipeline optimization',
        kind: 'SpaceSearchStep'
    },
    FeatureSelection: {
        name: 'Feature selection',
        kind: 'feature_selection'
        // drift_detection
    },
    PsudoLabeling: {
        name: 'Psudo labeling',
        kind: 'Psudo labeling'
    },
    PipelineReoptimization: {
        name: 'Pipeline re-optimization',
        kind: 'SpaceSearchStep'
    },
    Ensemble: {
        name: 'Ensemble',
        kind: 'EnsembleStep'
    }
};


export const MAX_FEATURES_OF_IMPORTANCES = 10;
