import {Steps, StepStatus} from "../constants";

export function getInitData() {

    return {
        steps: [
            {
                "name": Steps.Ensemble.name,
                "index": 0,
                "type":  Steps.Ensemble.type,
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
    }
}

export function sendFinishData(store, delay = 1000) {

    setTimeout(function () {
        store.dispatch(
            {
                type: 'stepFinished',
                payload: {
                    index: 0,
                    type: 'EnsembleStep',
                    extension: {
                        "weights": [0.1, 0.6, 0.3],
                        "scores": [0.1, 0.2, 0.3]
                    },
                    status: StepStatus.Finish,
                    datetime: ''
                }
            })
    }, delay);



}
