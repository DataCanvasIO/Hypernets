import {Steps, StepStatus} from "../constants";

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
                }
            }
        ]
    }
}

export function sendFinishData(store, delay = 1000) {

    setTimeout(function () {
        store.dispatch({
                type: 'stepFinished',
                payload: {
                    index: 0,
                    extension: {
                        importances: [
                            {name: 'id', importance: 0.1, dropped: true},
                            {name: 'id1', importance: 0.1, dropped: true},
                            {name: 'id2', importance: 0.8, dropped: false}
                        ]
                    },
                    status: StepStatus.Finish,
                    datetime: ''
                }
            })
    }, delay);

}
