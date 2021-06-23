import { Steps, StepStatus} from "../constants";

const CV = false;  // 控制模拟数据是否开启cv
const N_FOLDS = 3;

export function getInitData() {

    return {
        steps: [
            {
                "name": Steps.SpaceSearch.name,
                "index": 0,
                "type": Steps.SpaceSearch.type,
                "status": "wait",
                "configuration": {
                    "cv": false,
                    "name": "space_search",
                    "num_folds": null
                },
                "extension": {
                    trials: []
                },
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
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
                    status: {
                        reward: trialNoIndex/10,
                        noImprovedTrials: trialNoIndex + 2,
                        elapsedTime: 10000 + trialNoIndex * 5000
                    },
                    config: {
                        exceptedReward: 0.9,
                        maxNoImprovedTrials: 8,
                        maxElapsedTime: 100000,
                        direction: 'max'
                    }
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

        store.dispatch(
            {
                type: 'stepFinished',
                payload: {
                    index: 0,
                    type: 'SearchSpaceStep',
                    extension: {
                        input_features: [{"name": "age"}, {"name": "data"}],
                    },
                    status: StepStatus.Finish,
                    datetime: ''
                }
            }
        )

    }, 1000);
}, 500);

}


