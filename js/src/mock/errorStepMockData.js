import {ActionType, Steps, StepStatus} from "../constants";

export function getInitData() {

    return {
        steps: [
            {
                "name": Steps.PermutationImportanceSelection.name,
                "index": 0,
                "type": Steps.PermutationImportanceSelection.type,
                "status": "wait",
                "configuration": {
                    "cv": 100,
                },
                "start_datetime": 1626419128000,
                "end_datetime": null
            }
        ]
    }
}

export function sendFinishData(store, delay = 1000) {

    setTimeout(function () {
        store.dispatch({
                type: ActionType.StepBegin,
                payload: {
                    index: 0,
                    status: StepStatus.Process,
                    end_datetime: 1626419128
                }
            })
    }, delay);

    setTimeout(function () {
        store.dispatch({
            type: ActionType.StepError,
            payload: {
                index: 0,
                extension: {
                    reason: 'OutOfMemory'
                },
                status: StepStatus.Error,
                end_datetime: 1626419128
            }
        })
    }, delay * 2);
}
