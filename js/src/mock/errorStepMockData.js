import {ActionType, StepsKey, StepStatus} from "../constants";

export function getInitData() {

    return {
        steps: [
            {
                "name": StepsKey.PermutationImportanceSelection.name,
                "index": 0,
                "type": StepsKey.PermutationImportanceSelection.type,
                "status": "wait",
                "configuration": {
                    "cv": 100,
                }
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
                    datetime: ''
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
                datetime: ''
            }
        })
    }, delay * 2);
}
