export const experimentConfigMockData = {
    "steps": [{
        "index": 0,
        "name": "data_clean",
        "type": "DataCleanStep",
        "status": null,
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
            "random_state": 9527,
            "train_test_split_strategy": null
        },
        "extension": {},
        "start_datetime": null,
        "end_datetime": null
    }, {
        "index": 1,
        "name": "space_searching",
        "type": "SpaceSearchStep",
        "status": null,
        "configuration": {"cv": true, "name": "space_searching", "num_folds": 3},
        "extension": {},
        "start_datetime": null,
        "end_datetime": null
    }, {
        "index": 2,
        "name": "final_ensemble",
        "type": "EnsembleStep",
        "status": null,
        "configuration": {"ensemble_size": 20, "name": "final_ensemble", "scorer": null},
        "extension": {},
        "start_datetime": null,
        "end_datetime": null
    }, {
        "index": 3,
        "name": "space_searching",
        "type": "SpaceSearchStep",
        "status": null,
        "configuration": {"cv": true, "name": "space_searching", "num_folds": 3},
        "extension": {},
        "start_datetime": null,
        "end_datetime": null
    }, {
        "index": 4,
        "name": "final_ensemble",
        "type": "EnsembleStep",
        "status": null,
        "configuration": {"ensemble_size": 20, "name": "final_ensemble", "scorer": null},
        "extension": {},
        "start_datetime": null,
        "end_datetime": null
    }
    ]
};
