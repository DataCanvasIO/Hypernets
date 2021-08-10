import React from 'react';
import {showNotification} from "../util"
import {connect } from "react-redux";
import {ExperimentUI} from "./experiment"
import {ActionType} from "../constants";

const handleAction = (state, action, stepIndex, handler, actionType) => {
    const experimentConfig = state;

    var found = false;
    experimentConfig.steps.forEach((step, i, array) => {
        if (step.index === stepIndex) {
            found = true;
            // experimentConfig, action, stepIndexInArray
            handler(state, action, stepIndex)
        }
    });

    if (!found) {
        console.error(`Handler= ${actionType} index = ${action.stepIndex}  not found for update trial and action/state is :`);
        console.error(action);
        console.error(state);
    }
    return experimentConfig;

};

const handleProbaDensityLabelChange = (experimentConfig, action, stepIndexInArray) => {
    const  stepPayload = action.payload;
    experimentConfig.steps[stepIndexInArray].extension.selectedLabel = stepPayload.selectedLabel;
    return experimentConfig;
};


const handleFeatureImportanceChange = (experimentConfig, action, stepIndexInArray) => {
    const  stepPayload = action.payload;
    experimentConfig.steps[stepIndexInArray].extension.selectedTrialNo = stepPayload.selectedTrialNo;
    return experimentConfig;
};


const handleTrailFinish = (experimentConfig, action, stepIndexInArray) => {

    const {data: stepPayload} = action.payload;

    const searchStepExtension = experimentConfig.steps[stepIndexInArray].extension;
    if (searchStepExtension === undefined || searchStepExtension == null) {
        experimentConfig.steps[stepIndexInArray].extension = {}
    }
    const trials = experimentConfig.steps[stepIndexInArray].extension.trials;
    if (trials === undefined || trials === null) {
        experimentConfig.steps[stepIndexInArray].extension.trials = []
    }

    const trialData = {...stepPayload};

    experimentConfig.steps[stepIndexInArray].extension.earlyStopping = trialData.earlyStopping;
    experimentConfig.steps[stepIndexInArray].extension.maxTrials = trialData.maxTrials;  // persist maxTrials (does not use a special action)


    delete trialData.earlyStopping;

    experimentConfig.steps[stepIndexInArray].extension.trials.push(trialData);

    return experimentConfig;

};

const handleEarlyStopped = (experimentConfig, action, stepIndexInArray) => {
    const {payload} = action;
    experimentConfig.steps[stepIndexInArray].extension.earlyStopping = payload.data;
    return experimentConfig
};

const handleStepFinish = (experimentConfig, action, stepIndexInArray) => {

    const stepPayload = action.payload;
    const step = experimentConfig.steps[stepIndexInArray];
    if (step.type !== 'SpaceSearchStep') { // to avoid override 'trials'
        experimentConfig.steps[stepIndexInArray].extension = stepPayload.extension;
    } else {
        experimentConfig.steps[stepIndexInArray].extension.input_features = stepPayload.extension.input_features;
        experimentConfig.steps[stepIndexInArray].extension.features = stepPayload.extension.features;
    }
    // experimentConfig.steps[i].extension = stepPayload.extension;
    experimentConfig.steps[stepIndexInArray].status = stepPayload.status;
    experimentConfig.steps[stepIndexInArray].end_datetime = stepPayload.end_datetime;

    return experimentConfig;
};

const handleStepBegin = (experimentConfig, action, stepIndexInArray) => {
    const stepPayload = action.payload;
    const step = experimentConfig.steps[stepIndexInArray];
    step.status = stepPayload.status;
    const start_datetime = stepPayload.start_datetime;
    if(start_datetime !== undefined && start_datetime !== null){
        step.start_datetime = stepPayload.start_datetime
    }else {
        console.error("in step begin event but start_datetime is null ");
    }
    return experimentConfig;
};

const handleStepError = (experimentConfig, action, stepIndexInArray) => {
    const stepPayload = action.payload;

    const step = experimentConfig.steps[stepIndexInArray];

    step.status = stepPayload.status;
    const reason = stepPayload.extension.reason;
    if (reason !== null && reason !== undefined){
        showNotification(<span>
            {reason.toString()}
        </span>);
    }
    return experimentConfig;

};

// Map Redux state to component props
function mapStateToProps(state) {
    return {experimentData: state}
}

// Map Redux actions to component props
function mapDispatchToProps(dispatch) {
    return {dispatch}
}


// Reducer: Transform action to new state
export function experimentReducer(state, action) {
    const requestId = Math.random() * 10000;

    const {type, payload } = action;  // every action should has `type` and `payload` field
    console.info(`Rev action(${requestId}): `);
    console.info(action);
    console.info(`Rev state(${requestId}): `);
    console.info(state);

    let newState;
    if (type === ActionType.StepFinished) {
        const { index } = payload;
        newState  = handleAction(state, action, index, handleStepFinish, type);
    } else if (type === ActionType.StepBegin) {
        const { index } = payload;
        newState  = handleAction(state, action, index, handleStepBegin, type);
    } else if (type === ActionType.StepError) {
        const { index } = payload;
        newState  = handleAction(state, action, index, handleStepError, type);
    } else if (type === ActionType.ProbaDensityLabelChange) {
        const { stepIndex } = payload;
        newState  = handleAction(state, action, stepIndex, handleProbaDensityLabelChange, type);
    } else if (type === ActionType.FeatureImportanceChange) {
        const { stepIndex } = payload;
        newState  = handleAction(state, action, stepIndex, handleFeatureImportanceChange, type);
    } else if (type === ActionType.TrialFinished) {
        const { stepIndex } = payload;
        newState  = handleAction(state, action, stepIndex, handleTrailFinish, type);
    } else if (type === ActionType.EarlyStopped) {
        const { stepIndex } = payload;
        newState  = handleAction(state, action, stepIndex, handleEarlyStopped, type);
    } else {
        if(!type.startsWith('@@redux')){  // redux built-in action type
            console.error("Unseen action type: " + type);
        }
        newState = state
    }

    console.info(`Output new state(${requestId}): `);
    console.info(newState);
    return {...newState};
}


// Connected Component
export const ExperimentUIContainer = connect(
    mapStateToProps,
    mapDispatchToProps
)(ExperimentUI);


