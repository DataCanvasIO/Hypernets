import React from 'react';
import ReactDOM from 'react-dom';
import {ExperimentUI} from "./pages/experiment"
import {Steps} from "antd";
import {createStore} from "redux";
import {connect, Provider} from "react-redux";
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


const handleStepFinish = (state, action) => {
    const experimentConfig = state;
    const stepPayload = action.payload;
    var found = false;
    experimentConfig.steps.forEach((step, i, array) => {
        if (step.index === stepPayload.index) {
            if (step.type !== 'SpaceSearchStep') {
                experimentConfig.steps[i].extension = stepPayload.extension;
            } else {
                experimentConfig.steps[i].extension.input_features = stepPayload.extension.input_features;
            }
            // experimentConfig.steps[i].extension = stepPayload.extension;
            experimentConfig.steps[i].status = stepPayload.status;
            found = true;
        }
    });
    if (!found) {
        console.error("Step index = " + action.index + "not found for update step action/state is :");
        console.error(action);
        console.error(state);
    }

    return {...experimentConfig};

};

const handleProbaDensityLabelChanged = (state, action) => {
    const experimentConfig = state;
    const stepPayload = action.payload;
    const stepIndex = action.payload.stepIndex;
    var found = false;
    experimentConfig.steps.forEach((step, i, array) => {
        if (step.index === stepIndex) {
            experimentConfig.steps[i].extension.selectedLabel = stepPayload.selectedLabel;
            found = true;
        }
    });
    if (!found) {
        console.error("Step index = " + action.index + "not found for update selected label, action/state is :");
        console.error(action);
        console.error(state);
    }
    return {...experimentConfig};

};

const getNewTrialData = (trialNoIndex, isLatest) => {
    let models;
    if (CV) {
        models = Array.from({length: N_FOLDS}, (k, v) => v).map(
            i => {
                return {
                    reward: Math.random(),
                    fold: i,
                    importances: {'age': Math.random()}
                }
            }
        )
    } else {
        models = [{
            reward: 0.7,
            fold: 1,
            importances: {'age': Math.random()}
        }]
    }

    return {
        type: 'trialFinished',
        payload: {
            stepIndex: 3,
            trialData: {
                trialNo: trialNoIndex,
                hyperParams: {
                    max_depth: 10,
                    n_estimator: 100
                },
                models: models,
                reward: 0.7,
                elapsed: 100,
                metricName: 'auc',
                earlyStopping: {
                    conditionStatus: {
                        reward: 0.8,
                        noImprovedTrials: trialNoIndex + 2,
                        elapsedTime: 10000 + trialNoIndex * 5000
                    },
                    stopReason: {
                        condition: isLatest ? 'reward' : null,
                        value: isLatest ? 0.9 : null,
                    }
                }
            }
        }
    }
};

const handleTrailFinish = (state, action) => {
    const experimentConfig = state;
    const stepPayload = action.payload;

    var found = false;
    experimentConfig.steps.forEach((step, i, array) => {
        if (step.index === stepPayload.stepIndex) {
            found = true;
            const searchStepExtension = experimentConfig.steps[i].extension;
            if (searchStepExtension === undefined || searchStepExtension == null) {
                experimentConfig.steps[i].extension = {}
            }
            const trials = experimentConfig.steps[i].extension.trials;
            if (trials === undefined || trials === null) {
                experimentConfig.steps[i].extension.trials = []
            }
            experimentConfig.steps[i].extension.trials.push(stepPayload.trialData);
        }
    });

    if (!found) {
        console.error("Step index = " + action.stepIndex + "not found for update trial and action/state is :");
        console.error(action);
        console.error(state);
    }

    return {...experimentConfig};

};

// Reducer
function experimentReducer(state, action) {
    // Transform action to state
    const {type} = action;
    if (type === 'experimentData') {
        return {experimentData: action}
    } else if (type === 'stepFinished') {
        const newState = handleStepFinish(state, action);
        console.info("new state");
        console.info(newState);
        return newState;
    } else if (type === 'probaDensityLabelChange') {
        const newState = {...state};
        return handleProbaDensityLabelChanged(newState, action)
    } else if (type === 'trialFinished') {
        return handleTrailFinish(state, action)
    } else {
        return state;
    }
}

// Store
export const store = createStore(experimentReducer, experimentConfigData(d => d));


export function renderPipelineMatrixBundle(ele, experimentData) {
    ReactDOM.render(
        <Provider store={store}>
            <ExperimentUIContainer experimentData={experimentData}/>
        </Provider>,
        ele
    );
}


// Map Redux state to component props
function mapStateToProps(state) {
    return {experimentData: state}
}

// Map Redux actions to component props
function mapDispatchToProps(dispatch) {
    return {dispatch}
}

// Connected Component
const ExperimentUIContainer = connect(
    mapStateToProps,
    mapDispatchToProps
)(ExperimentUI);


// todo dev
ReactDOM.render(
    // <MyComponent percentage={percentage} />,
    <Provider store={store}>
        <ExperimentUIContainer experimentData={store.getState()}/>
    </Provider>,
    document.getElementById('root')
);


setTimeout(function () {
    store.dispatch(
        {
            type: 'stepFinished',
            payload: {
                index: 0,
                type: 'DataCleanStep',
                extension: {unselected_features: [{name: "id", reason: 'unknown'}]},
                status: StepStatus.Finish,
                datetime: ''
            }
        }
    )
}, 100);


setTimeout(function () {
    store.dispatch(
        {
            type: 'stepFinished',
            payload: {
                index: 1,
                type: 'MulticollinearityDetectStep',
                extension: {unselected_features: [{"removed": "age", "reserved": "data"}]} ,
                status: StepStatus.Finish,
                datetime: ''
            }
        }
    )
}, 100);
// {unselected_features: [{name: "id", reason: 'unknown'}]}

setTimeout(function () {
    store.dispatch(
        {
            type: 'stepFinished',
            payload: {
                index: 2,
                type: 'DriftedDetectedStep',
                extension:  {
                    drifted_features_auc: [
                        {
                            feature: "id",
                            score: 0.6
                        }, {
                            feature: "default",
                            score: 0.6
                        }, {
                            feature: "education",
                            score: 0.6
                        }
                    ],
                    removed_features_in_epochs: [
                        {
                            epoch: 0,
                            removed_features: [
                                {
                                    feature: 'education',
                                    importance: 0.1,
                                }
                            ]
                        },
                        {
                            epoch: 1,
                            removed_features: [
                                {
                                    feature: 'id',
                                    importance: 0.11,
                                }
                            ]
                        }
                    ]
                },
                status: StepStatus.Finish,
                datetime: ''
            }
        }
    )
}, 100);



var fakeTrialNo = 0;
let trialInterval;
setTimeout(function () {
    trialInterval = setInterval(function () {
        fakeTrialNo = fakeTrialNo + 1;
        if (fakeTrialNo <= 5) {
            store.dispatch(
                getNewTrialData(fakeTrialNo, fakeTrialNo === 5)
            )
        } else {
            clearInterval(trialInterval);
        }
    }, 200);
}, 500);

setTimeout(function () {
    store.dispatch(
        {
            type: 'stepFinished',
            payload: {
                index: 3,
                type: 'SearchSpaceStep',
                extension: {
                    input_features: [{"name": "age"}, {"name": "data"}],
                },
                status: StepStatus.Finish,
                datetime: ''
            }
        }
    )
}, 3000);


setTimeout(function () {
    store.dispatch(
        {
            type: 'stepFinished',
            payload: {
                index: 4,
                type: StepsKey.FeatureSelection.type,
                extension: {
                    importances: [
                        {name: 'id', importance: 0.1, dropped: true},
                        {name: 'id1', importance: 0.1, dropped: true},
                        {name: 'id2', importance: 0.1, dropped: true}
                    ]
                },
                status: StepStatus.Finish,
                datetime: ''
            }
        })
}, 4000);

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


setTimeout(function () {
    store.dispatch(
        {
            type: 'stepFinished',
            payload: {
                index: 5,
                type: StepsKey.PsudoLabeling.name,
                extension: {
                    probabilityDensity: {
                        yes: {
                            gaussian: {
                                X: [-1,0,1],
                                probaDensity: [0.1, 0.2]
                            }
                        },
                        no: {
                            gaussian: {
                                X: [-1,0,1],
                                probaDensity: [0.9, 0.8]
                            }
                        },
                    },
                    samples: {
                        1: 1000,
                        2: 2000
                    },
                    selectedLabel: "yes"
                },
                status: StepStatus.Finish,
                datetime: ''
            }
        })
}, 5000);


setTimeout(function () {
    store.dispatch(
        {
            type: 'stepFinished',
            payload: {
                index: 7,
                type: 'EnsembleStep',
                extension: {
                    "weights": [0.1, 0.6, 0.3],
                    "lifting": [0.1, 0.2, 0.3]
                },
                status: StepStatus.Finish,
                datetime: ''
            }
        })
}, 6000);

