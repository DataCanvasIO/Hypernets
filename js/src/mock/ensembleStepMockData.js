import {ActionType, StepStatus} from "../constants";
import {EnsembleStep, DaskEnsembleStep} from "../components/steps";

export function getInitData(distribution=false) {

    const  cls = distribution ? DaskEnsembleStep:EnsembleStep;
    return {
        steps: [
            {
                "name": cls.getDisplayName(),
                "index": 0,
                "type":  cls.getTypeName(),
                "status": StepStatus.Wait,
                "configuration": {
                    "ensemble_size": 20,
                    "name": "final_ensemble",
                    "scorer": null
                },
                "extension": null,
                "start_datetime": 1626419128,
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
                    start_datetime: 1626419128,
                    status: StepStatus.Process
                }
            })
    }, delay);

    setTimeout(function () {
        store.dispatch({
            type: 'stepFinished',
            payload: {
                index: 0,
                status: StepStatus.Finish,
                end_datetime: 1626419128,
                extension: {
                    features: {
                        inputs: ['name', 'age'],
                        outputs: null,
                        increased: null,
                        reduced: null
                    },
                    scores: [0.1, 0.3],
                    weights: [0.1, 0.3],
                }
            }
        })
    }, delay * 2);

}
