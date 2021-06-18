import React from 'react';
import ReactDOM from 'react-dom';
import { ExperimentSummary } from './pages/experimentSummary'
import { StepsKey, StepStatus } from "./constants";
import { Dataset } from './pages/dataset'
import { experimentReducer, ExperimentUIContainer } from './pages/experimentRedux'
import { getInitData, sendFinishData } from './mock/errorStepMockData'
import { connect, Provider } from "react-redux";
import { createStore } from "redux";


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
                    "name": "data_clean",
                    "random_state": 9527,
                    "train_test_split_strategy": null,
                    "data_cleaner_args": {
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
                "index": 2,
                "type": StepsKey.CollinearityDetection.type,
                "status": "wait",
                "configuration": {
                    collinearity_detection: true
                },
                "extension": null,
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
            }, {
                "name": StepsKey.FeatureSelection.type,
                "index": 3,
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
            },
            {
                "name": StepsKey.DriftDetection.name,
                "index": 5,
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
                "index": 6,
                "type": "SpaceSearchStep",
                "status": "wait",
                "configuration": {
                    "cv": CV,
                    "name": "space_search",
                    "num_folds": N_FOLDS
                },
                "extension": {
                    trials: []
                },
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
            }, {
                "name": StepsKey.PermutationImportanceSelection.type,
                "index": 7,
                "type": StepsKey.PermutationImportanceSelection.type,
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
                "index": 8,
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
                "name": StepsKey.SpaceSearch.type,
                "index": 9,
                "type": StepsKey.SpaceSearch.type,
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
                "index": 10,
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

const d = [{"name": "data_clean", "index": 7, "type": "DataCleanStep", "status": null, "configuration": {"cv": true, "data_cleaner_args": {}, "name": "data_clean", "random_state": 9527, "train_test_split_strategy": null, "data_cleaner_params": {"nan_chars": null, "correct_object_dtype": true, "drop_constant_columns": true, "drop_label_nan_rows": true, "drop_idness_columns": true, "drop_columns": null, "drop_duplicated_columns": false, "reduce_mem_usage": false, "int_convert_to": "float"}}, "extension": null, "start_datetime": null, "end_datetime": null}, {"name": "drift_detection", "index": 7, "type": "DriftDetectStep", "status": null, "configuration": {"min_features": 10, "name": "drift_detection", "num_folds": 5, "remove_shift_variable": true, "remove_size": 0.1, "threshold": 0.7, "variable_shift_threshold": 0.7}, "extension": null, "start_datetime": null, "end_datetime": null}, {"name": "space_searching", "index": 7, "type": "SpaceSearchStep", "status": null, "configuration": {"cv": true, "name": "space_searching", "num_folds": 3}, "extension": null, "start_datetime": null, "end_datetime": null}, {"name": "final_ensemble", "index": 7, "type": "EnsembleStep", "status": null, "configuration": {"ensemble_size": 20, "name": "final_ensemble"}, "extension": null, "start_datetime": null, "end_datetime": null}]


export function renderDatasetSummary(data, domElement){
    ReactDOM.render(
        <Dataset data={data}/>,
        domElement
    );
}

export function renderExperimentSummary(data, domElement){
    ReactDOM.render(
        <ExperimentSummary experimentData={data}/>,
        domElement
    );
}


export function renderExperimentProcess(experimentData, domElement) {
    const store = createStore(experimentReducer, experimentData);
    ReactDOM.render(
        <Provider store={store}>
            <ExperimentUIContainer/>
        </Provider>,
        domElement
    );
    return store
}

// renderDatasetSummary(experimentConfigData(v => v), document.getElementById('root'));
// renderExperimentSummary({steps: d}, document.getElementById('root'));

// const store = renderExperimentProcess(getInitData(), document.getElementById('root'));
//
// sendFinishData(store);
//

// setTimeout(function () {
//     store.dispatch(
//         {
//             type: 'stepFinished',
//             payload: {
//                 index: 0,
//                 type: 'DataCleanStep',
//                 extension: {unselected_reason: {"id": 'unknown'}},
//                 status: StepStatus.Finish,
//                 datetime: ''
//             }
//         }
//     )
// }, 500);
//
//
// setTimeout(function () {
//     store.dispatch(
//         {
//             type: 'stepFinished',
//             payload: {
//                 index: 1,
//                 type: 'MulticollinearityDetectStep',
//                 extension: {unselected_features: [{"removed": "age", "reserved": "data"}]} ,
//                 status: StepStatus.Finish,
//                 datetime: ''
//             }
//         }
//     )
// }, 100);
//
//
// setTimeout(function () {
//     store.dispatch(
//         {
//             type: 'stepFinished',
//             payload: {
//                 index: 2,
//                 type: 'DriftedDetectedStep',
//                 extension:  {
//                     drifted_features_auc: [
//                         {
//                             feature: "id",
//                             score: 0.6
//                         }, {
//                             feature: "default",
//                             score: 0.6
//                         }, {
//                             feature: "education",
//                             score: 0.6
//                         }
//                     ],
//                     removed_features_in_epochs: [
//                         {
//                             epoch: 0,
//                             removed_features: [
//                                 {
//                                     feature: 'education',
//                                     importance: 0.1,
//                                 }
//                             ]
//                         },
//                         {
//                             epoch: 1,
//                             removed_features: [
//                                 {
//                                     feature: 'id',
//                                     importance: 0.11,
//                                 }
//                             ]
//                         }
//                     ]
//                 },
//                 status: StepStatus.Finish,
//                 datetime: ''
//             }
//         }
//     )
// }, 100);
//
//
// const getNewTrialData = (trialNoIndex, isLatest) => {
//     let models;
//     if (CV) {
//         models = Array.from({length: N_FOLDS}, (k, v) => v).map(
//             i => {
//                 return {
//                     fold: i,
//                     importances: {'age': Math.random()}
//                 }
//             }
//         )
//     } else {
//         models = [{
//             fold: null,
//             importances: {'age': Math.random()}
//         }]
//     }
//
//     return {
//         type: 'trialFinished',
//         payload: {
//             stepIndex: 3,
//             trialData: {
//                 trialNo: trialNoIndex,
//                 hyperParams: {
//                     max_depth: 10,
//                     n_estimator: 100
//                 },
//                 models: models,
//                 reward: 0.7,
//                 elapsed: 100,
//                 metricName: 'auc',
//                 earlyStopping: {
//                     status: {
//                         reward: trialNoIndex/10,
//                         noImprovedTrials: trialNoIndex + 2,
//                         elapsedTime: 10000 + trialNoIndex * 5000
//                     }, config: {
//                         exceptedReward: 0.9,
//                         maxNoImprovedTrials: 8,
//                         maxElapsedTime: 100000,
//                         direction: 'max'
//                     }
//                 }
//             }
//         }
//     }
// };
//
// var fakeTrialNo = 0;
// let trialInterval;
// setTimeout(function () {
//     trialInterval = setInterval(function () {
//         fakeTrialNo = fakeTrialNo + 1;
//         if (fakeTrialNo <= 5) {
//             store.dispatch(
//                 getNewTrialData(fakeTrialNo, fakeTrialNo === 5)
//             )
//         } else {
//             clearInterval(trialInterval);
//         }
//     }, 1000);
// }, 500);
//
// setTimeout(function () {
//     store.dispatch(
//         {
//             type: 'stepFinished',
//             payload: {
//                 index: 3,
//                 type: 'SearchSpaceStep',
//                 extension: {
//                     input_features: [{"name": "age"}, {"name": "data"}],
//                 },
//                 status: StepStatus.Finish,
//                 datetime: ''
//             }
//         }
//     )
// }, 3000);


// setTimeout(function () {
//     store.dispatch(
//         {
//             type: 'stepFinished',
//             payload: {
//                 index: 4,
//                 type: StepsKey.FeatureSelection.type,
//                 extension: {
//                     importances: [
//                         {name: 'id', importance: 0.1, dropped: true},
//                         {name: 'id1', importance: 0.1, dropped: true},
//                         {name: 'id2', importance: 0.1, dropped: true}
//                     ]
//                 },
//                 status: StepStatus.Finish,
//                 datetime: ''
//             }
//         })
// }, 4000);

// setTimeout(function () {
//     store.dispatch(
//         {
//             type: 'stepFinished',
//             payload: {
//                 index: 5,
//                 type: 'SpaceSearchStep',
//                 extension: {unselected_features: [{"removed": "age", "reserved": "data"}]},
//                 status: StepStatus.Finish,
//                 datetime: ''
//             }
//         }
//     )
// }, 200);

// setTimeout(function () {
//     store.dispatch(
//         {
//             type: 'stepFinished',
//             payload: {
//                 index: 5,
//                 type: StepsKey.PsudoLabeling.name,
//                 extension: {
//                     probabilityDensity: {
//                         yes: {
//                             gaussian: {
//                                 X: [-1,0,1],
//                                 probaDensity: [0.1, 0.2]
//                             }
//                         },
//                         no: {
//                             gaussian: {
//                                 X: [-1,0,1],
//                                 probaDensity: [0.9, 0.8]
//                             }
//                         },
//                     },
//                     samples: {
//                         1: 1000,
//                         2: 2000
//                     },
//                     selectedLabel: "yes"
//                 },
//                 status: StepStatus.Finish,
//                 datetime: ''
//             }
//         })
// }, 5000);


// setTimeout(function () {
//     store.dispatch(
//         {
//             type: 'stepFinished',
//             payload: {
//                 index: 7,
//                 type: 'EnsembleStep',
//                 extension: {
//                     "weights": [0.1, 0.6, 0.3],
//                     "scores": [0.1, 0.2, 0.3]
//                 },
//                 status: StepStatus.Finish,
//                 datetime: ''
//             }
//         })
// }, 6000);
//


