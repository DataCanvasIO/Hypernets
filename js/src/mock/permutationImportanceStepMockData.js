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
                type: 'stepFinished',
                payload: {
                    index: 0,
                    extension: {
                        importances: [
                            {name: 'id', importance: 0.1, dropped: true},
                            {name: 'id1', importance: 0.1, dropped: true},
                            {name: 'id2', importance: 0.8, dropped: false},
                            {name: 'id3', importance: 0.8, dropped: false},
                            {name: 'id4', importance: 0.8, dropped: false},
                            {name: 'id5', importance: 0.8, dropped: false},
                            {name: 'id6', importance: 0.8, dropped: false},
                            {name: 'id7', importance: 0.8, dropped: false},
                            {name: 'id8', importance: 0.8, dropped: false},
                            {name: 'id9', importance: 0.8, dropped: false},
                            {name: 'id10', importance: 0.8, dropped: false},
                            {name: 'id11', importance: 0.8, dropped: false},
                            {name: 'id12', importance: 0.8, dropped: false},
                            {name: 'id13', importance: 0.8, dropped: false},
                            {name: 'id14', importance: 0.3, dropped: false},
                            {name: 'id15', importance: 0.3, dropped: false},
                        ]
                    },
                    status: StepStatus.Finish,
                    end_datetime: 1626419128
                }
            })
    }, delay);

}
