export function getInitData(CV = false) {

    return {
        "steps": [{
            "index": 0,
            "name": "data_clean",
            "type": "DataCleanStep",
            "status": "wait",
            "configuration": {
                "cv": true,
                "data_cleaner_args": {
                    "nan_chars": null,
                    "correct_object_dtype": true,
                    "drop_constant_columns": true,
                    "drop_label_nan_rows": true,
                    "drop_idness_columns": true,
                    "drop_columns": null,
                    "drop_duplicated_columns": false,
                    "reduce_mem_usage": false,
                    "int_convert_to": "float"
                },
                "name": "data_clean",
                "train_test_split_strategy": null
            },
            "extension": {},
            "start_datetime": null,
            "end_datetime": null
        }, {
            "index": 1,
            "name": "multicollinearity_detection",
            "type": "MulticollinearityDetectStep",
            "status": "wait",
            "configuration": {"name": "multicollinearity_detection"},
            "extension": {},
            "start_datetime": null,
            "end_datetime": null
        }, {
            "index": 2,
            "name": "drift_detection",
            "type": "DriftDetectStep",
            "status": "wait",
            "configuration": {
                "min_features": 6,
                "name": "drift_detection",
                "num_folds": 5,
                "remove_shift_variable": true,
                "remove_size": 0.2,
                "threshold": 0.47,
                "variable_shift_threshold": 0.51
            },
            "extension": {},
            "start_datetime": null,
            "end_datetime": null
        }, {
            "index": 3,
            "name": "feature_selection",
            "type": "FeatureImportanceSelectionStep",
            "status": "wait",
            "configuration": {
                "name": "feature_selection",
                "number": null,
                "quantile": null,
                "strategy": "threshold",
                "threshold": 0.001
            },
            "extension": {},
            "start_datetime": null,
            "end_datetime": null
        }, {
            "index": 4,
            "name": "space_searching",
            "type": "SpaceSearchStep",
            "status": "wait",
            "configuration": {"cv": true, "name": "space_searching", "num_folds": 3},
            "extension": {},
            "start_datetime": null,
            "end_datetime": null
        }, {
            "index": 5,
            "name": "ensemble",
            "type": "EnsembleStep",
            "status": "wait",
            "configuration": {"ensemble_size": 20, "name": "ensemble", "scorer": null},
            "extension": {},
            "start_datetime": null,
            "end_datetime": null
        }, {
            "index": 6,
            "name": "pseudo_labeling",
            "type": "PseudoLabelStep",
            "status": "wait",
            "configuration": {
                "estimator_builder_name": "ensemble",
                "name": "pseudo_labeling",
                "proba_quantile": null,
                "proba_threshold": 0.9,
                "resplit": false,
                "sample_number": null,
                "strategy": "threshold"
            },
            "extension": {},
            "start_datetime": null,
            "end_datetime": null
        }, {
            "index": 7,
            "name": "feature_reselection",
            "type": "PermutationImportanceSelectionStep",
            "status": "wait",
            "configuration": {
                "estimator_size": 10,
                "name": "feature_reselection",
                "number": null,
                "quantile": null,
                "scorer": "make_scorer(roc_auc_score, needs_threshold=True)",
                "strategy": "threshold",
                "threshold": 0.1
            },
            "extension": {},
            "start_datetime": null,
            "end_datetime": null
        }, {
            "index": 8,
            "name": "two_stage_searching",
            "type": "SpaceSearchStep",
            "status": "wait",
            "configuration": {"cv": true, "name": "two_stage_searching", "num_folds": 3},
            "extension": {},
            "start_datetime": null,
            "end_datetime": null
        }, {
            "index": 9,
            "name": "final_ensemble",
            "type": "EnsembleStep",
            "status": "wait",
            "configuration": {"ensemble_size": 20, "name": "final_ensemble", "scorer": null},
            "extension": {},
            "start_datetime": null,
            "end_datetime": null
        }]
    }

}


export function sendFinishData(store, delay = 1000) {

}