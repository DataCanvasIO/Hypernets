// Append display name to experimentData, the name only required by frontend
import {Steps, TWO_STAGE_SUFFIX} from "../constants";


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
        // check type
        var found = false;
        var stepMetaData = null;
        // 1. find meta data
        Object.keys(Steps).forEach(k => {
            if (Steps[k].type === stepType) {
                found = true;
                stepMetaData = Steps[k];
            }
        });

        if (found === false) {
            console.error("Unseen step type: " + stepType);
            return;
        }
        add(stepsCounter, stepType);
        const stepCount = stepsCounter[stepType];

        // 2. get step ui title
        const stepName = stepMetaData.name;
        let stepTitle;
        if (stepCount > 1) {
            stepTitle = stepName + TWO_STAGE_SUFFIX
        } else {
            stepTitle = stepName;
        }

        stepData['displayName'] = stepTitle;
        stepData['meta'] = stepMetaData;
    }

    return experimentData;
}