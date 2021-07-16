import {ActionType, Steps, StepStatus} from "../constants";

export function getInitData() {

    return {
        steps: [
            {
                "name": Steps.Ensemble.name,
                "index": 0,
                "type":  Steps.Ensemble.type,
                "status": StepStatus.Wait,
                "configuration": {
                    "ensemble_size": 20,
                    "name": "final_ensemble",
                    "scorer": null
                },
                "extension": null,
                "start_datetime": 1626419128000,
                "end_datetime": null
            }
        ]
    }
}

export function sendFinishData(store, delay = 2000) {

    setTimeout(function () {
        store.dispatch({
                type: ActionType.StepBegin,
                payload: {
                    index: 0,
                    status: StepStatus.Process
                }
            })
    }, delay);

    setTimeout(function () {
        store.dispatch({
            type: 'stepFinished',
            payload: {
                index: 0,
                type: 'EnsembleStep',
                extension: {
                },
                status: StepStatus.Skip,
                end_datetime: 1626419128
            }
        })
    }, delay * 2);

}
