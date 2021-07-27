// Append display name to experimentData, the name only required by frontend
import {Steps, TWO_STAGE_SUFFIX} from "../constants";
import {getStepComponent} from "./steps";
import {isEmpty, notEmpty} from "../util";


export function prepareExperimentData(experimentData) {
    const steps = {...experimentData}.steps; // must clone a copy to avoid modify origin data
    const stepsCounter = {};
    const add = (counter, key) => {
        const v = counter[key];
        if(v === undefined || v === null){
            counter[key] = 1
        }else{
            counter[key] = v + 1;
        }
    };

    for(const stepData of steps){
        const stepType = stepData.type;
        // 1. find meta data
        const CompCls = getStepComponent(stepType);
        if(isEmpty(CompCls)){
            console.error("Unseen step type: " + stepType);
            return null;
        }

        const displayName = CompCls.getDisplayName();

        add(stepsCounter, stepType);
        const stepCount = stepsCounter[stepType];

        // 2. get step ui title
        let stepTitle;
        if (stepCount > 1) {
            stepTitle = displayName + TWO_STAGE_SUFFIX
        } else {
            stepTitle = displayName;
        }

        stepData['displayName'] = stepTitle;
    }

    return experimentData;
}