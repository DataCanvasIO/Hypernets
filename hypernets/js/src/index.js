import React from 'react';
import ReactDOM from 'react-dom';
import reportWebVitals from './reportWebVitals';
import { ExperimentUI } from "./pages/experiment"
import {Steps} from "antd";


const experimentConfigData = {
    steps: [
        {
            kind: 'data_cleaning',
            start_datetime: "2020-10-10 10:10:10",
            end_datetime: "2020-10-10 10:20:10",
            configuration: {
                nan_chars: null,
                correct_object_dtype: true,
                drop_constant_columns: true,
                drop_label_nan_rows: true,
                drop_idness_columns: true,
                replace_inf_values: true,
                drop_columns: null,
                drop_duplicated_columns: true,
                reduce_mem_usage: true,
                int_convert_to: 'float'
            },
            extension: {
                dropped_columns: [
                    {
                        "name": "id",
                        "reason": "idness"
                    },{
                        "name": "default",
                        "reason": "constant"
                    },{
                        "name": "pdays",
                        "reason": "duplicate"
                    }
                ]
            }
        },{
            kind: 'collinearity_detection',
            start_datetime: "2020-10-10 10:10:10",
            end_datetime: "2020-10-10 10:20:10",
            configuration: {
                collinearity_detection: true
            },
            extension: {
                dropped_columns: [
                    {
                        "removed": "id",
                        "reserved": "namme"
                    },{
                        "removed": "default",
                        "reserved": "education"
                    }
                ]
            }
        },{
            kind: 'drift_detection',
            start_datetime: "2020-10-10 10:10:10",
            end_datetime: "2020-10-10 10:20:10",
            configuration: {
                drift_detection: true,
                remove_shift_variable: true,
                variable_shift_threshold: 0.7,
                threshold: 0.7,
                remove_size: 0.1,
                min_features: 10,
                num_folds: 5,
            },
            extension: {
                drifted_features_auc: [
                    {
                        feature: "id",
                        score: 0.6
                    },{
                        feature: "default",
                        score: 0.6
                    },{
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
    ]
};


ReactDOM.render(
  <React.StrictMode>
    <ExperimentUI configData={experimentConfigData} />
  </React.StrictMode>,
  document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
