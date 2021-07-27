import React from 'react';
import  { useState } from  "react";

import "antd/dist/antd.css";
import { Steps as AntdSteps, Progress, Card } from 'antd';

import { StepStatus, ProgressBarStatus } from '../constants'
import { getStepComponent } from '../components/steps';
import { prepareExperimentData } from '../components/prepare';

const { Step } = AntdSteps;


export function ExperimentUI ({experimentData, dispatch} ) {

    const [currentStepIndex , setCurrentStepIndex] = useState(0);
    const stepTabComponents = [];
    const stepContentComponents = [];

    const steps = prepareExperimentData(experimentData).steps;
    // experimentData.steps;

    const getStepUIStatus = (stepStatus)=> {
        if(stepStatus === StepStatus.Skip){
            return StepStatus.Finish
        } else {
            return stepStatus;
        }
    };


    const getProcessBarStatus  = (steps) => {
        var processFinish = true;
        for (var step of steps) {
            const stepUIStatus = getStepUIStatus(step.status);

            if (stepUIStatus === StepStatus.Error) {
                return ProgressBarStatus.Exception
            }
            if(stepUIStatus !== StepStatus.Finish){  // all step is finish so the ProcessBar is succeed
                processFinish = false;
            }
        }
        if(processFinish){
            return ProgressBarStatus.Success
        }else{
            return ProgressBarStatus.Active
        }
    };

    const getProcessPercentage = (steps) => {
        // 1. find last finished step index
        var lastFinishedStepIndex = -1;
        for(var i = steps.length - 1; i >= 0 ; i--){
            if( getStepUIStatus(steps[i].status) === StepStatus.Finish){
                lastFinishedStepIndex = i;
                break;
            }
        }
        // 2. last finished step index / total step
        return (((lastFinishedStepIndex + 1) / steps.length) * 100).toFixed(0);
    };

    for(var stepData of steps){
        const stepType = stepData.type;
        const stepTitle = stepData.displayName;
        const stepUIStatus = getStepUIStatus(stepData.status);

        const StepComp = getStepComponent(stepType);
        if(StepComp === null || StepComp === undefined){
             // showNotification("Unknown step type");
             console.error("Internal error, unhandled step type: " + stepType);
             return ;
        }

        const stepStyle = { marginLeft: 20, marginRight: 20 };
        stepTabComponents.push(
            <Step status={stepUIStatus} title={stepTitle} key={`step_${stepTitle}`} style={stepStyle} />
        );

        stepContentComponents.push(
            <StepComp stepData={stepData} dispatch={dispatch} key={`step_comp_${stepTitle}`}  />
        );
    }

    const onStepChange = (c) => {
        setCurrentStepIndex(c);
    };

    return <Card title="Experiment progress" bordered={false} style={{ width: '85%', alignContent: 'center' }}>
            <Progress percent={  getProcessPercentage(steps) } status={ getProcessBarStatus (steps)} />
            <div style={ { overflowX: 'auto', marginTop:10 }}>
                <AntdSteps
                    type="navigation"
                    size="small"
                    current={currentStepIndex}
                    onChange={onStepChange}
                >
                    {stepTabComponents}
                </AntdSteps>
            </div>
            {stepContentComponents[currentStepIndex]}
        </Card>
}