export const datasetMockData_1 = {
    "experimentType": "compete",
    "target": {
        "name": "y",
        "taskType": "binary",
        "freq": 713,
        "unique": 2,
        "missing": 0,
        "mean": null,
        "min": null,
        "max": null,
        "stdev": null,
        "dataType": "str"
    },
    "targetDistribution": {
        "no": 713,
        "yes": 87
    },
    "datasetShape": {
        "X_train": [
            800,
            17
        ],
        "y_train": [
            800
        ],
        "X_eval": [],
        "y_eval": [],
        "X_test": []
    },
    "featureDistribution": {
        "nContinuous": 8,
        "nText": 0,
        "nDatetime": 0,
        "nCategorical": 9,
        "nLocation": 0,
        "nOthers": 0
    }
};

export const datasetMockData = {
    "experimentType": "compete",
    "target": {
        "name": "y",
        "taskType": "regression",
        "freq": null,
        "unique": 207,
        "missing": 0,
        "mean": 22.796534653465343,
        "min": 5.0,
        "max": 50.0,
        "stdev": 9.332147158711562,
        "dataType": "float"
    },
    "targetDistribution": {
        "count": [122, 66, 65, 42, 32, 25, 18, 17, 9, 7],
        "region": [[18.5, 23.0], [23.0, 27.5], [14.0, 18.5], [9.5, 14.0], [27.5, 32.0], [32.0, 36.5], [5.0, 9.5], [45.5, 50.0], [41.0, 45.5], [36.5, 41.0]]
    },
    "datasetShape": {"X_train": [404, 13], "y_train": [404], "X_eval": [], "y_eval": [], "X_test": []},
    "featureDistribution": {
        "nContinuous": 13,
        "nText": 0,
        "nDatetime": 0,
        "nCategorical": 0,
        "nLocation": 0,
        "nOthers": 0
    }
};
