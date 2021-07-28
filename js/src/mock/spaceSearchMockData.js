import {ActionType, Steps, StepStatus} from "../constants";

const CV = false;  // 控制模拟数据是否开启cv
const N_FOLDS = 3;

export function getInitData() {

    return {
        steps: [
            {
                "name": Steps.SpaceSearch.name,
                "index": 0,
                "type": Steps.SpaceSearch.type,
                "status": StepStatus.Wait,
                "configuration": {
                    "cv": true,
                    "name": "space_searching",
                    "num_folds": 3,
                    "earlyStopping": {
                        "enable": false,
                        "exceptedReward": 1,
                        "maxNoImprovedTrials": 10,
                        "timeLimit": 60000,
                        "mode": "max"
                    }
                },
                "extension": {
                    trials: [],
                    "earlyStopping": {
                        "bestReward": null,
                        "bestTrialNo": null,
                        "counterNoImprovementTrials": null,
                        "triggered": null,
                        "triggeredReason": null,
                        "elapsedTime": null
                    },
                    features: {
                        inputs: ['name', 'age'],
                        outputs: ['name', 'age'],
                        increased: [],
                        reduced: []
                    },
                },
                "start_datetime": 1626419128,
                "end_datetime": null
            }
        ]
    }
}

const getNewTrialData = (trialNoIndex, isLatest) => {
    let models;
    if (CV) {
        models = Array.from({length: N_FOLDS}, (k, v) => v).map(
            i => {
                return {
                    fold: i,
                    importances: [{ name: 'age', imp: Math.random()}]
                }
            }
        )
    } else {
        models = [{
            fold: null,
            importances: [{ name: 'age', imp: Math.random()}]
        }]
    }

    return {
        type: 'trialFinished',
        payload: {
            stepIndex: 0,
            data: {
                trialNo: trialNoIndex,
                maxTrials: 6,
                hyperParams: {
                    max_depth: 10,
                    n_estimator: 100
                },
                models: models,
                reward: Math.random() * 10,
                elapsed: Math.random() * 100,
                metricName: 'auc',
                earlyStopping: {
                    bestReward: Math.random(),
                    bestTrialNo: 1,
                    counterNoImprovementTrials: trialNoIndex + 2,
                    triggered: false,
                    triggeredReason: null,
                    elapsedTime: 10000 + trialNoIndex * 5000
                }
            }
        }
    }
};


export function sendFinishData(store, delay = 1000) {

    var fakeTrialNo = 0;
    let trialInterval;
    setTimeout(function () {
        trialInterval = setInterval(function () {
            fakeTrialNo = fakeTrialNo + 1;
            if (fakeTrialNo <= 5) {
                const d = getNewTrialData(fakeTrialNo, fakeTrialNo === 5);
                store.dispatch(
                    d
                )
            } else {
                clearInterval(trialInterval);
            }
        }, 1000);
    }, 500);

    setTimeout(function () {
        store.dispatch(
            {
                type: ActionType.EarlyStopped,
                payload: {
                    stepIndex: 0,
                    data:{
                        bestReward: 0.9,
                        bestTrialNo: 1,
                        counterNoImprovementTrials: 10,
                        triggered: true,
                        triggeredReason: null,
                        elapsedTime: 10000
                    }
                }
            }
        )
    }, 10000);

    setTimeout(function () {
        store.dispatch(
            {
                type: 'stepFinished',
                payload: {
                    index: 0,
                    extension: {
                        input_features: [{"name": "age"}, {"name": "data"}],
                        features: {
                            inputs: ['name', 'age'],
                            outputs: ['name_1', 'name'],
                            increased: ['name_1'],
                            reduced: ['age']
                        }
                    },
                    status: StepStatus.Finish,
                    end_datetime: 1626519128
                }
            }
        )
    }, 11000);

}


