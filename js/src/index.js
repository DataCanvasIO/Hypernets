import React from 'react';
import ReactDOM from 'react-dom';
import { ExperimentUI } from "./pages/experiment"
import {Steps} from "antd";
import {createStore} from "redux";
import {connect, Provider} from "react-redux";

const experimentConfigData_  = (handler) =>{
    const pd =  {
        type: 'experimentData',
        steps: [
            {
                kind: 'data_cleaning',
                index: 0,
                start_datetime: "2020-10-10 10:10:10",
                end_datetime: "2020-10-10 10:20:10",
                status: 'process',
                configuration: {
                    nan_chars: null,
                    correct_object_dtype: true,
                    drop_constant_columns: true,
                    drop_label_nan_rows: true,
                    drop_idness_columns: true,
                    replace_inf_values: true,
                    drop_columns: null,
                    drop_duplicated_columns: true,
                    reduce_mem_usage: true,
                    int_convert_to: 'float'
                },
                extension: {
                    dropped_columns: []
                    // dropped_columns: [
                    //     {
                    //         "name": "id",
                    //         "reason": "idness"
                    //     },{
                    //         "name": "default",
                    //         "reason": "constant"
                    //     },{
                    //         "name": "pdays",
                    //         "reason": "duplicate"
                    //     }
                    // ]
                }
            },{
                kind: 'collinearity_detection',
                start_datetime: "2020-10-10 10:10:10",
                status: 'wait',
                end_datetime: "2020-10-10 10:20:10",
                configuration: {
                    collinearity_detection: true
                },
                extension: {
                    dropped_columns: [
                        {
                            "removed": "id",
                            "reserved": "namme"
                        },{
                            "removed": "default",
                            "reserved": "education"
                        }
                    ]
                }
            },{
                kind: 'drift_detection',
                status: 'wait',
                start_datetime: "2020-10-10 10:10:10",
                end_datetime: "2020-10-10 10:20:10",
                configuration: {
                    drift_detection: true,
                    remove_shift_variable: true,
                    variable_shift_threshold: 0.7,
                    threshold: 0.7,
                    remove_size: 0.1,
                    min_features: 10,
                    num_folds: 5,
                },
                extension: {
                    drifted_features_auc: [
                        {
                            feature: "id",
                            score: 0.6
                        },{
                            feature: "default",
                            score: 0.6
                        },{
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
                }
            },{
                kind: 'pipeline_optimization',
                status: 'wait',
                start_datetime: "2020-10-10 10:10:10",
                end_datetime: "2020-10-10 10:20:10",
                configuration: {
                    cv: true,
                    num_folds: 3
                },
                extension: {
                    trials: [
                        {
                            trial_no: 1,
                            reward_metric: 'auc',
                            reward_score: 0.7,
                            elapsedTime: 1300,
                            models: [
                                {
                                    fold_no: 1,
                                    reward_metric: 'auc',
                                    reward_score: 0.7,
                                    feature_importance: [
                                        {
                                            feature: 'age',
                                            importance: 0.1
                                        }
                                    ]
                                }
                            ],
                            hyper_params: {
                                max_depth: 10,
                                n_estimators: 100
                            }
                        },
                    ],
                    input_features: ['col_1', 'col_2', 'col_3', 'col_4', 'col_5']
                }
            }
        ]
    };
    return handler(pd);
};

const experimentConfigData  = (handler) =>{
    const pd = {
        steps: [
            {
        "name": 'DataCleanStep',
        "index": 0,
        "type": "DataCleanStep",
        "status": "process",
        "configuration": {
            "cv": true,
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
        "name": 'DriftDetectStep',
        "index": 1,
        "type": "DriftDetectStep",
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
        "name": 'SpaceSearchStep',
        "index": 2,
        "type": "SpaceSearchStep",
        "status": "wait",
        "configuration": {
            "cv": true,
            "name": "space_search",
            "num_folds": 3
        },
        "extension": null,
        "start_datetime": "2020-11-11 22:22:22",
        "end_datetime": "2020-11-11 22:22:22"
    }, {
        "name": 'DataCleanStep',
        "index": 3,
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
    }]
    };

    return handler(pd);
};



const handleStepFinish = (state, action) => {
    const experimentConfig = state;
    const stepPayload = action.payload;
    var found = false;
    experimentConfig.steps.forEach((step, i, array) => {
        if(step.index === stepPayload.index){
            if(step.type !== 'SpaceSearchStep'){
                experimentConfig.steps[i].extension = stepPayload.extension;
            }
            // experimentConfig.steps[i].extension = stepPayload.extension;
            experimentConfig.steps[i].status = stepPayload.status;
            found = true;
        }
    });
    if(!found){
        console.error("Step index = " + action.index + "not found for update step action/state is :");
        console.error(action);
        console.error(state);
    }

    return {...experimentConfig};

};


const getNewTrialData = (trialNoIndex) => {
    return {
        type: 'trialFinished',
        payload: {
            stepIndex: 2,
            trialData: {
                trialNo: trialNoIndex,
                hyperParams: {
                    max_depth: 10,
                    n_estimator: 100
                },
                models: [
                    {
                        reward: 0.7,
                        fold: 1,
                        importances: {'age': Math.random()}
                    }
                ],
                avgReward: 0.7,
                elapsed: 100,
                metricName: 'auc'
            }
        }
    }
};

const handleTrailFinish = (state, action) => {
    const experimentConfig = state;
    const stepPayload = action.payload;

    var found = false;
    experimentConfig.steps.forEach((step, i, array) => {
        if(step.index === stepPayload.stepIndex){
            found = true;
            const searchStepExtension = experimentConfig.steps[i].extension;
            if(searchStepExtension === undefined || searchStepExtension == null){
                experimentConfig.steps[i].extension = {}
            }
            const trials = experimentConfig.steps[i].extension.trials;
            if (trials === undefined || trials === null){
                experimentConfig.steps[i].extension.trials = []
            }
            experimentConfig.steps[i].extension.trials.push(stepPayload.trialData);
        }
    });

    if(!found){
        console.error("Step index = " + action.stepIndex + "not found for update trial and action/state is :");
        console.error(action);
        console.error(state);
    }

    return {...experimentConfig};

};

// Reducer
function experimentReducer(state , action) {
    // Transform action to state
    const {type} = action;
    if(type === 'experimentData'){
        return {experimentData: action}
    }else if (type === 'stepFinished'){
        const newState =  handleStepFinish(state, action);
        console.info("new state");
        console.info(newState);
        return newState;
    } else if (type === 'trialFinished') {
        return handleTrailFinish(state, action)
    }
    else{
        return state;
    }
}

// Store
export const store = createStore(experimentReducer, experimentConfigData( d => d ));


export function renderPipelineMatrixBundle(ele, experimentData){
    ReactDOM.render(
        <Provider store={store}>
            <ExperimentUIContainer  experimentData={experimentData}/>
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
            type:'stepFinished',
            payload: {
                index: 0,
                type: 'DataCleanStep',
                extension: {unselected_features: [{name: "id", reason: 'unknown'}]},
                status: 'finish',
                datetime: ''
            }
        }
    )
}, 200);




setTimeout(function () {
    store.dispatch(
        {
            type:'stepFinished',
            payload: {
                index: 1,
                type: 'DataCleanStep',
                extension: {unselected_features: [{"removed": "age", "reserved": "data"}] },
                status: 'finish',
                datetime: ''
            }
        }
    )
}, 400);

var fakeTrialNo = 0;
let trialInterval;
setTimeout(function () {
    trialInterval = setInterval(function () {
        fakeTrialNo = fakeTrialNo + 1;
        if(fakeTrialNo <= 5){
            store.dispatch(
                getNewTrialData(fakeTrialNo)
            )
        }else{
            clearInterval(trialInterval);
        }
    }, 1000);
}, 1000);

setTimeout(function () {
    store.dispatch(
        {
            type:'stepFinished',
            payload: {
                index: 2,
                type: 'SearchSpaceStep',
                extension: {
                    unselected_features: [{"removed": "age", "reserved": "data"}],
                },
                status: 'finish',
                datetime: ''
            }
        }
    )
}, 11000);

setTimeout(function () {
    store.dispatch(
        {
            type: 'stepFinished',
            payload: {
                index: 3,
                type: 'EnsembleStep',
                extension: {
                    "weights": [0.1, 0.6, 0.3],
                    "lifting": [0.1, 0.2, 0.3]
                },
                status: 'finish',
                datetime: ''
            }
        })
}, 12000);
