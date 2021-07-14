import React from 'react';
import ReactDOM from 'react-dom';
import { ExperimentSummary } from './pages/experimentSummary'
import { Steps } from "./constants";
import { Dataset } from './pages/dataset'
import { experimentReducer, ExperimentUIContainer } from './pages/experimentRedux'
import { getInitData, sendFinishData } from './mock/ensembleStepMockData'
import { datasetMockData, datasetMockDataClassification } from './mock/plotDatasetMockData'
import { experimentConfigMockData } from './mock/experimentConfigMockData'
import { Provider } from "react-redux";
import { createStore } from "redux";


const CV = true;  // 控制模拟数据是否开启cv
const N_FOLDS = 3;


const experimentConfigData = (handler) => {
    const pd = {
        steps: [
            {
                "name": Steps.CollinearityDetection.name,
                "index": 2,
                "type": Steps.CollinearityDetection.type,
                "status": "wait",
                "configuration": {
                    collinearity_detection: true
                },
                "extension": null,
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
            }, {
                "name": Steps.FeatureSelection.type,
                "index": 3,
                "type": Steps.FeatureSelection.type,
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
                "name": Steps.DriftDetection.name,
                "index": 5,
                "type": Steps.DriftDetection.type,
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
                "name": Steps.PermutationImportanceSelection.type,
                "index": 7,
                "type": Steps.PermutationImportanceSelection.type,
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
                "name": Steps.PsudoLabeling.type,
                "index": 8,
                "type": Steps.PsudoLabeling.type,
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
                "name": Steps.SpaceSearch.type,
                "index": 9,
                "type": Steps.SpaceSearch.type,
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
            }
        ]
    };

    return handler(pd);
};

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

// ----------------------------Test Experiment UI----------------------------------------
// const store = renderExperimentProcess(getInitData(), document.getElementById('root'));
// sendFinishData(store);
// --------------------------------------------------------------------------------------

// ----------------------------Test Dataset----------------------------------------
// renderDatasetSummary(datasetMockDataClassification, document.getElementById('root'));
// --------------------------------------------------------------------------------------

// ----------------------------Test Experiment Summary----------------------------------------
// renderExperimentSummary(experimentConfigMockData, document.getElementById('root'));
// --------------------------------------------------------------------------------------

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

// setTimeout(function () {
//     store.dispatch(
//         {
//             type: 'stepFinished',
//             payload: {
//                 index: 4,
//                 type: Steps.FeatureSelection.type,
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
//                 type: Steps.PsudoLabeling.name,
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


