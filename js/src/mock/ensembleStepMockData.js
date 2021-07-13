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
                "start_datetime": "2020-11-11 22:22:22",
                "end_datetime": "2020-11-11 22:22:22"
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
                datetime: ''
            }
        })
    }, delay * 2);

}
