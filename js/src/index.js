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
        "kind": "DataCleanStep",
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
        "kind": "DriftDetectStep",
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
        "kind": "SpaceSearchStep",
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
        "kind": "EnsembleStep",
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

const getNewTrialData = (trialNoIndex) => {
    return {
            trialNo: trialNoIndex,
                hyperParams: {
                max_depth: 10,
                    n_estimator: 100
            },
            models: [
                {
                    reward: 0.7,
                    fold: 1,
                    importances: [
                        {name: 'age', importance: 0.1}
                    ]
                }
            ],
            avgReward: 0.7,
            elapsed: 100,
            metricName: 'auc'
    }

};

// Reducer
function experimentReducer(state={} , action) {
    // Transform action to state
    const {type} = action;
    if(type === 'experimentData'){
        return {experimentData: action}
    }else if (type === 'stepFinished'){
        return {newStepData: action.data};
    } else if (type === 'trialFinished') {
        return {newTrialData: action.data};
    }
    else{
        return state;
    }
}

// Store
export const store = createStore(experimentReducer);


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
    return state
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
        <ExperimentUIContainer
            experimentData={experimentConfigData( d => d )}
        />
    </Provider>,
    document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals

//
setTimeout(function () {
    store.dispatch(
        experimentConfigData(d => {
            d.steps[0].extension =  { unselected_features: ['id']} ;
            d.steps[0].status =  'finish' ;
            return {
                type: 'stepFinished',
                data: d.steps[0]
            }
        })
    )
}, 2000);

setTimeout(function () {
    store.dispatch(
        experimentConfigData(d => {
            d.steps[1].extension =  { unselected_features: [{"removed": "age", "reserved": "data"}] } ;
            d.steps[1].status =  'finish' ;
            return {
                type: 'stepFinished',
                data: d.steps[1]
            }
        })
    )
}, 4000);




var fakeTrialNo = 0;
let trialInterval;
setTimeout(function () {
    trialInterval = setInterval(function () {
        fakeTrialNo = fakeTrialNo + 1;
        store.dispatch(
            {
                type: 'trialFinished',
                data: getNewTrialData(fakeTrialNo)
            }
        )
    }, 1000);
}, 6000);


// setTimeout(function () {
//     store.dispatch(
//         experimentConfigData(d => {
//             d.steps[2].extension =  {
//                 "estimator": null,
//             };
//             d.steps[2].status =  'finish' ;
//             return {
//                 type: 'stepFinished',
//                 data: d.steps[2]
//             }
//         })
//     )
// }, 12000);
//
//
// setTimeout(function () {
//     store.dispatch(
//         experimentConfigData(d => {
//             d.steps[3].extension =  {
//                 "estimator": null,
//                 "weights": [0.1, 0.6, 0.3],
//                 "lifting": [0.1, 0.2, 0.3]
//             };
//             d.steps[3].status =  'finish' ;
//             return {
//                 type: 'stepFinished',
//                 data: d.steps[3]
//             }
//         })
//     )
// }, 14000);
