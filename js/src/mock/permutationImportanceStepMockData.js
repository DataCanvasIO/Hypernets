import {StepsKey, StepStatus} from "../constants";

export function getInitData() {

    return {
        steps: [
            {
                "name": StepsKey.PermutationImportanceSelection.name,
                "index": 0,
                "type": StepsKey.PermutationImportanceSelection.type,
                "status": "process",
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
