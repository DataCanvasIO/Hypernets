import React from 'react';
import {ExperimentUI} from "./experiment"
import {connect, Provider} from "react-redux";

const handleStepFinish = (state, action) => {
    const experimentConfig = state;
    const stepPayload = action.payload;
    var found = false;
    experimentConfig.steps.forEach((step, i, array) => {
        if (step.index === stepPayload.index) {
            if (step.type !== 'SpaceSearchStep') {
                experimentConfig.steps[i].extension = stepPayload.extension;
            } else {
                experimentConfig.steps[i].extension.input_features = stepPayload.extension.input_features;
            }
            // experimentConfig.steps[i].extension = stepPayload.extension;
            experimentConfig.steps[i].status = stepPayload.status;
            found = true;
        }
    });
    if (!found) {
        console.error("Step index = " + action.index + "not found for update step action/state is :");
        console.error(action);
        console.error(state);
    }
    return {...experimentConfig};
};

const handleProbaDensityLabelChanged = (state, action) => {
    const experimentConfig = state;
    const stepPayload = action.payload;
    const stepIndex = action.payload.stepIndex;
    var found = false;
    experimentConfig.steps.forEach((step, i, array) => {
        if (step.index === stepIndex) {
            experimentConfig.steps[i].extension.selectedLabel = stepPayload.selectedLabel;
            found = true;
        }
    });
    if (!found) {
        console.error("Step index = " + action.index + "not found for update selected label, action/state is :");
        console.error(action);
        console.error(state);
    }
    return {...experimentConfig};

};

const handleTrailFinish = (state, action) => {
    const experimentConfig = state;
    const stepPayload = action.payload;

    var found = false;
    experimentConfig.steps.forEach((step, i, array) => {
        if (step.index === stepPayload.stepIndex) {
            found = true;
            const searchStepExtension = experimentConfig.steps[i].extension;
            if (searchStepExtension === undefined || searchStepExtension == null) {
                experimentConfig.steps[i].extension = {}
            }
            const trials = experimentConfig.steps[i].extension.trials;
            if (trials === undefined || trials === null) {
                experimentConfig.steps[i].extension.trials = []
            }
            experimentConfig.steps[i].extension.trials.push(stepPayload.trialData);
        }
    });

    if (!found) {
        console.error("Step index = " + action.stepIndex + "not found for update trial and action/state is :");
        console.error(action);
        console.error(state);
    }

    return {...experimentConfig};

};

// Map Redux state to component props
function mapStateToProps(state) {
    return {experimentData: state}
}

// Map Redux actions to component props
function mapDispatchToProps(dispatch) {
    return {dispatch}
}


// Reducer
export function experimentReducer(state, action) {
    // Transform action to state
    const {type} = action;
    console.info("Rev state: ");
    console.info(state);

    console.info("Rev action: " );
    console.info(action);

    if (type === 'experimentData') {
        return {experimentData: action}
    } else if (type === 'stepFinished') {
        const newState = handleStepFinish(state, action);
        console.info("new state");
        console.info(newState);
        return newState;
    } else if (type === 'probaDensityLabelChange') {
        const newState = {...state};
        return handleProbaDensityLabelChanged(newState, action)
    } else if (type === 'trialFinished') {
        return handleTrailFinish(state, action)
    } else {
        return state;
    }
}


// Connected Component
export const ExperimentUIContainer = connect(
    mapStateToProps,
    mapDispatchToProps
)(ExperimentUI);


