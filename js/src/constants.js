export const StepsKey = {
    DataCleaning: {
        name: 'Data cleaning',
        type: 'DataCleanStep'
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
        type: 'SpaceSearchStep'
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
    ReSpaceSearch: {
        name: 'Pipeline re-optimization',
        type: 'ReSpaceSearch'
    },
    Ensemble: {
        name: 'Ensemble',
        type: 'EnsembleStep'
    }
};


export const MAX_FEATURES_OF_IMPORTANCES = 10;
