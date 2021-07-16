import React from 'react';
import ReactDOM from 'react-dom';
import { ExperimentSummary } from './pages/experimentSummary'
import { Steps } from "./constants";
import { Dataset } from './pages/dataset'
import { experimentReducer, ExperimentUIContainer } from './pages/experimentRedux'
import { getInitData, sendFinishData } from './mock/driftDetectionStepMockData.js'
import { datasetMockData, datasetMockDataClassification } from './mock/plotDatasetMockData.js'
import { experimentConfigMockData } from './mock/experimentConfigMockData'
import { Provider } from "react-redux"
import { createStore } from "redux"
import { Result, Button } from 'antd'


export function renderDatasetSummary(data, domElement){
    ReactDOM.render(
        <Dataset data={data}/>,
        domElement
    );
}

export function renderExperimentSummary(data, domElement){
    ReactDOM.render(
        <ExperimentSummary experimentData={data}/>,
        domElement
    );
}


export function renderExperimentProcess(experimentData, domElement) {
    const store = createStore(experimentReducer, experimentData);
    ReactDOM.render(
        <Provider store={store}>
            <ExperimentUIContainer/>
        </Provider>,
        domElement
    );
    return store
}

export function renderLossState(domElement) {
    ReactDOM.render(
        <Result
            status="warning"
            title="The experiment state data was lost"
            subTitle={"Maybe you refreshed the page before the experiment finished, you may be able to create the widget by running the appropriate cells. "}
        />,
        domElement
    );
}


// ----------------------------Test Experiment UI----------------------------------------
const store = renderExperimentProcess(getInitData(), document.getElementById('root'));
sendFinishData(store);
// --------------------------------------------------------------------------------------

// ----------------------------Test Dataset----------------------------------------
// renderDatasetSummary(datasetMockDataClassification, document.getElementById('root'));
// --------------------------------------------------------------------------------------

// ----------------------------Test Experiment Summary----------------------------------------
// renderExperimentSummary(experimentConfigMockData, document.getElementById('root'));
// --------------------------------------------------------------------------------------


// ----------------------------Test Render loss state -----------------------------------
// renderLossState(document.getElementById('root'));
// --------------------------------------------------------------------------------------
