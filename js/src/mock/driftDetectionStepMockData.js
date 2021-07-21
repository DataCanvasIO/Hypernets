import {Steps, StepStatus} from "../constants";

export function getInitData(CV=false) {

    return {
        steps: [
            {
                "name": Steps.DriftDetection.name,
                "index": 0,
                "type": Steps.DriftDetection.type,
                "status": StepStatus.Wait,
                "configuration": {
                    "min_features": 10,
                    "name": "drift_detection",
                    "num_folds": 5,
                    "remove_shift_variable": true,
                    "remove_size": 0.1,
                    "threshold": 0.7,
                    "variable_shift_threshold": 0.7
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
            {
                type: 'stepFinished',
                payload: {
                    index: 0,
                    status: StepStatus.Finish,
                    end_datetime: 1626449128,
                    extension:  {
                        features: {
                            inputs: ['name', 'age'],
                            outputs: ['name_1', 'name'],
                            increased: ['name_1'],
                            reduced: ['age']
                        },
                        drifted_features_auc: [
                            {
                                feature: "id",
                                score: 0.6
                            }, {
                                feature: "default",
                                score: 0.6
                            }, {
                                feature: "education",
                                score: 0.6
                            }
                        ],
                        removed_features_in_epochs: [
                            {
                                epoch: 0,
                                removed_features: [
                                    {
                                        feature: 'education',
                                        importance: 0.1,
                                    }
                                ]
                            },
                            {
                                epoch: 1,
                                removed_features: [
                                    {
                                        feature: 'id',
                                        importance: 0.11,
                                    }
                                ]
                            }
                        ]
                    }
                }

            }
        )
    }, delay);
}
