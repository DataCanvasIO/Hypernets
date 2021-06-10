import React from 'react';
import ReactDOM from 'react-dom';
import { ExperimentSummary } from './pages/experimentSummary'
import Block from 'react-blocks';
import {StepsKey, StepStatus} from "./constants";

const CV = true;  // 控制模拟数据是否开启cv
const N_FOLDS = 3;


const experimentConfigData = (handler) => {
    const pd = {
        steps: [
            {
                "name": StepsKey.DataCleaning.name,
                "index": 0,
                "type": StepsKey.DataCleaning.type,
                "status": "process",
                "configuration": {
                    "cv": CV,
                    "data_cleaner_args": {},
                    "name": "data_clean",
                    "random_state": 9527,
                    "train_test_split_strategy": null,
                    "data_cleaner_params": {
                        "nan_chars": null,
                        "correct_object_dtype": true,
                        "drop_constant_columns": true,
                        "drop_label_nan_rows": true,
                        "drop_idness_columns": true,
                        "replace_inf_values": null,
                        "drop_columns": null,
                        "drop_duplicated_columns": false,
                        "reduce_mem_usage": false,
                        "int_convert_to": "float"
                    }
                },
                "extension": null,
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
            }, {
                "name": StepsKey.CollinearityDetection.name,
                "index": 1,
                "type": StepsKey.CollinearityDetection.type,
                "status": "wait",
                "configuration": {
                    collinearity_detection: true
                },
                "extension": null,
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
            },
            {
                "name": StepsKey.DriftDetection.name,
                "index": 2,
                "type": StepsKey.DriftDetection.type,
                "status": "wait",
                "configuration": {
                    "min_features": 10,
                    "name": "drift_detection",
                    "num_folds": 5,
                    "remove_shift_variable": true,
                    "remove_size": 0.1,
                    "threshold": 0.7,
                    "variable_shift_threshold": 0.7
                },
                "extension": null,
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
            }, {
                "name": StepsKey.SpaceSearch.type,
                "index": 3,
                "type": "SpaceSearchStep",
                "status": "wait",
                "configuration": {
                    "cv": CV,
                    "name": "space_search",
                    "num_folds": N_FOLDS,
                    "earlyStopping": {
                        "exceptedReward": 0.9,
                        "maxNoImprovedTrials": 8,
                        "maxElapsedTime": 100000,
                        "direction": 'max'
                    }
                },
                "extension": {
                    trials: [],
                    "earlyStopping": {
                        "conditionStatus": {
                            "reward": 0,
                            "maxNoImprovedTrials": 0,
                            "elapsedTime": 0
                        },
                        "stopReason": {
                            "condition": null,
                            "value": null
                        }
                    }
                },
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
            }, {
                "name": StepsKey.FeatureSelection.type,
                "index": 4,
                "type": StepsKey.FeatureSelection.type,
                "status": "wait",
                "configuration": {
                    "feature_reselection": true,
                    "estimator_size": 10,
                    "threshold": 0.00001
                },
                "extension": {
                    importances: []
                },
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
            },{
                "name": StepsKey.PsudoLabeling.type,
                "index": 5,
                "type": StepsKey.PsudoLabeling.type,
                "status": "wait",
                "configuration": {
                    "proba_threshold": 0.8,
                    "resplit": false,
                    "strategy": "s1"
                },
                "extension": {},
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
            }, {
                "name": StepsKey.ReSpaceSearch.type,
                "index": 6,
                "type": StepsKey.ReSpaceSearch.type,
                "status": "wait",
                "configuration": {
                    "cv": CV,
                    "name": "space_search",
                    "num_folds": N_FOLDS,
                    "earlyStopping": {
                        "exceptedReward": 0.9,
                        "maxNoImprovedTrials": 8,
                        "maxElapsedTime": 100000,
                        "direction": 'max'
                    }
                },
                "extension": {
                    trials: [],
                    "earlyStopping": {
                        "conditionStatus": {
                            "reward": 0,
                            "maxNoImprovedTrials": 0,
                            "elapsedTime": 0
                        },
                        "stopReason": {
                            "condition": null,
                            "value": null
                        }
                    }
                },
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
            },
            {
                "name": StepsKey.Ensemble.type,
                "index": 7,
                "type": "EnsembleStep",
                "status": "wait",
                "configuration": {
                    "ensemble_size": 20,
                    "name": "final_ensemble",
                    "scorer": null
                },
                "extension": null,
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
            }
        ]
    };

    return handler(pd);
};


ReactDOM.render(
    <ExperimentSummary experimentData={experimentConfigData(v => v)}/>,
    document.getElementById('root')
);

