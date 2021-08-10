import {Steps, StepStatus} from "../constants";
import {PseudoLabelStep, DaskPseudoLabelStep} from "../components/steps";

export function getInitData(distribution=false) {
    const cls = distribution ? DaskPseudoLabelStep: PseudoLabelStep;
    return {
        steps: [
            {
                "name": cls.getDisplayName(),
                "index": 0,
                "type": cls.getTypeName(),
                "status": "wait",
                "configuration": {
                    "proba_threshold": 0.8,
                    "resplit": false,
                    "strategy": "s1"
                },
                "extension": {},
                "start_datetime": 1626419128,
                "end_datetime": null
            }
        ]
    }
}

export function sendFinishData(store, delay = 1000) {

    setTimeout(function () {
        store.dispatch(
            { type: 'stepFinished',
            payload: {
                    index: 0,
                    status: StepStatus.Finish,
                    end_datetime: 1626419728,
                    extension: {
                        probabilityDensity: {
                            yes: {
                                gaussian: {
                                    X: [-1,0,1],
                                    probaDensity: [0.1, 0.2]
                                }
                            },
                            no: {
                                gaussian: {
                                    X: [-1,0,1],
                                    probaDensity: [0.9, 0.8]
                                }
                            },
                        },
                        samples: {
                            'yes': 1000,
                            'no': 2000
                        },
                        selectedLabel: "yes",
                        features: {
                            inputs: ['name', 'age'],
                            outputs: ['name_1', 'name'],
                            increased: ['name_1'],
                            reduced: ['age']
                        }
                    }

            }
        })
    }, delay);
}
