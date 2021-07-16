import {Steps, StepStatus} from "../constants";

export function getInitData(CV=false) {
    return {
        steps: [
            {
                "name": Steps.CollinearityDetection.name,
                "index": 0,
                "type": Steps.CollinearityDetection.type,
                "status": StepStatus.Wait,
                "configuration": {
                    collinearity_detection: true
                },
                "extension": null,
                "start_datetime": 1626419128,
                "end_datetime": null
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
                    extension: {unselected_features: [{"removed": "age", "reserved": "data"}]} ,
                    status: StepStatus.Finish,
                    datetime: 1626519128
                }
            }
        )
    }, delay);
}
