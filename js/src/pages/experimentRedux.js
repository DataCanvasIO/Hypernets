import React from 'react';
import {showNotification} from "../util"
import {connect } from "react-redux";
import {ExperimentUI} from "./experiment"
import {ActionType} from "../constants";



const handleProbaDensityLabelChanged = (state, action) => {
    const experimentConfig = {...state};
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
            experimentConfig.steps[i].end_datetime = stepPayload.end_datetime;
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

const handleStepBegin = (state, action) => {
    const experimentConfig = state;
    const stepPayload = action.payload;

    var found = false;
    experimentConfig.steps.forEach((step, i, array) => {
        if (step.index === stepPayload.index) {
            found = true;
            step.status = stepPayload.status
        }
    });

    if (!found) {
        console.error("Step index = " + action.index + "not found for update step begin action/state is :");
        console.error(action);
        console.error(state);
    }

    return {...experimentConfig};

};


const handleStepError = (state, action) => {
    const experimentConfig = state;
    const stepPayload = action.payload;

    var found = false;
    experimentConfig.steps.forEach((step, i, array) => {
        if (step.index === stepPayload.index) {
            found = true;
            step.status = stepPayload.status;
            const reason = stepPayload.extension.reason;
            if (reason !== null && reason !== undefined){
                showNotification(<span>
                    {reason.toString()}
                </span>);
            }
        }
    });

    if (!found) {
        console.error("Step index = " + action.index + "not found for update step error action/state is :");
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

    let newState;
    if (type === ActionType.ExperimentData) {
        return {experimentData: action}
    } else if (type === ActionType.StepFinished) {
        newState = handleStepFinish(state, action);
    } else if (type === ActionType.StepBegin) {
        newState =  handleStepBegin(state, action)
    } else if (type === ActionType.StepError) {
        newState =  handleStepError(state, action)
    } else if (type === ActionType.ProbaDensityLabelChange) {
        newState =  handleProbaDensityLabelChanged(state, action)
    } else if (type === ActionType.TrialFinished) {
        newState =  handleTrailFinish(state, action)
    }else {
        newState =  state;
    }
    console.info("Output new state :" );
    console.info(newState);
    return newState;
}


// Connected Component
export const ExperimentUIContainer = connect(
    mapStateToProps,
    mapDispatchToProps
)(ExperimentUI);


