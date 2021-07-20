import {Steps, StepStatus} from "../constants";

export function getInitData() {

    return {
        steps: [
            {
                "name": Steps.FeatureGeneration.name,
                "index": 0,
                "type": Steps.FeatureGeneration.type,
                "status": "wait",
                "configuration": {
                    collinearity_detection: true,
                    collinearity_aaaa_bbbbbb_ccc: "collinearity_aaaa_bbbbbb_ccccollinearity_aaaa_bbbbbb_ccc"
                },
                "extension": {},
                "start_datetime": 1626419128000,
                "end_datetime": null
            },
        ]
    }
}

export function sendFinishData(store, delay = 1000) {

    setTimeout(function () {
        store.dispatch({
                type: 'stepFinished',
                payload: {
                    index: 0,
                    status: StepStatus.Finish,
                    end_datetime: 1626419128,
                    extension: {
                        outputFeatures: [
                            {
                                name: 'id_name',
                                parentFeatures: ['id', 'name'],
                                primitive: 'add'
                            },
                            {
                                name: 'id_name_aa',
                                parentFeatures: null,
                                primitive: 'add'
                            }
                        ]
                    }
                }
            })
    }, delay);

}
