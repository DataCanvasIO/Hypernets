import React from 'react';
import  { useState } from  "react";

import "antd/dist/antd.css";
import {  Steps as AntdSteps, Progress, Card} from 'antd';
import "antd/dist/antd.css";

import {
    CollinearityDetectionStep,
    DriftDetectionStep,
    EnsembleStep,
    PseudoLabelStep,
    DataCleaningStep,
    FinalTrainStep,
    GeneralImportanceSelectionStep,
    FeatureGenerationStep
} from '../components/steps'

import { Steps, StepStatus, TWO_STAGE_SUFFIX } from '../constants'
import { PipelineOptimizationStep} from '../components/pipelineSearchStep'

const { Step } = AntdSteps;


const ProgressBarStatus = {
    Success: 'success',
    Exception : 'exception',
    Normal : 'normal',
    Active : 'active',
};

export function ExperimentUI ({experimentData, dispatch} ) {

    const [currentStepIndex , setCurrentStepIndex] = useState(0);
    const stepTabComponents = [];
    const stepContentComponents = [];
    const steps = experimentData.steps;


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
        for(var i = experimentData.steps.length - 1; i >= 0 ; i--){
            if( getStepUIStatus(steps[i].status) === StepStatus.Finish){
                lastFinishedStepIndex = i;
                break;
            }
        }
        // 2. last finished step index / total step
        return (((lastFinishedStepIndex + 1) / steps.length) * 100).toFixed(0);
    };

    const add = (counter, key) => {
        const v = counter[key];
        if(v === undefined || v === null){
            counter[key] = 1
        }else{
            counter[key] = v + 1;
        }
    };

    // step type count
    const stepsCounter = {};

    experimentData.steps.forEach(stepData=> {
        const stepType = stepData.type;

        // check type
        var found = false;
        var stepMetaData = null;
        Object.keys(Steps).forEach(k => {
            if(Steps[k].type === stepType){
                found = true;
                stepMetaData = Steps[k];
            }
        });

        if(found === false){
            console.error("Unseen step type: " + stepType);
            return ;
        }
        add(stepsCounter, stepType);
        const stepCount = stepsCounter[stepType];

        // set step ui title
        const stepName = stepMetaData.name;
        let stepTitle ;
        if (stepCount > 1){
            stepTitle = stepName + TWO_STAGE_SUFFIX
        }else {
            stepTitle = stepName;
        }

        // set step status
        const stepUIStatus = getStepUIStatus(stepData.status);

        const getComponent = (stepType) => {
            if(stepType  === Steps.DataCleaning.type){
                return DataCleaningStep ;
            }else if(stepType  === Steps.FeatureGeneration.type){
                return FeatureGenerationStep ;
            } else if(stepType  === Steps.CollinearityDetection.type){
                return CollinearityDetectionStep;
            } else if(stepType  === Steps.DriftDetection.type){
                return DriftDetectionStep ;
            } else if(stepType  === Steps.SpaceSearch.type){
                return PipelineOptimizationStep;
            } else if(stepType  === Steps.FeatureSelection.type){
                return GeneralImportanceSelectionStep;
            }  else if(stepType  === Steps.PermutationImportanceSelection.type){
                return GeneralImportanceSelectionStep;
            } else if(stepType  === Steps.PsudoLabeling.type){
                return PseudoLabelStep
            } else if(stepType  === Steps.FinalTrain.type){
                return FinalTrainStep ;
            } else if(stepType  === Steps.Ensemble.type){
                return EnsembleStep ;
            } else {
                console.error("Internal error, unhandled step type: " + stepType);
                return ;
            }
        };

        const StepComp = getComponent(stepType);
        if(StepComp === null || StepComp === undefined){
             // showNotification("Unknown step type");
             console.error("Internal error, unhandled step type: " + stepType);
             return ;
        }

        const stepStyle = {marginLeft: 20, marginRight: 20};
        stepTabComponents.push(
            <Step status={stepUIStatus} title={stepTitle} key={`step_${stepTitle}`} style={stepStyle} />
        );

        stepContentComponents.push(
            <StepComp stepData={stepData} dispatch={dispatch} key={`step_comp_${stepTitle}`} configTip={stepMetaData.configTip} />
        );
    });

    const onStepChange = (c) => {
        setCurrentStepIndex(c);
    };

    return <Card title="Experiment progress" bordered={false} style={{ width: '100%' }}>
            <Progress percent={  getProcessPercentage(steps) } status={ getProcessBarStatus (steps)} />
            <div style={ {width: '100%', overflowX: 'auto', marginTop:10 }}>
                <AntdSteps
                    type="navigation"
                    size="small"
                    current={currentStepIndex}
                    onChange={onStepChange}
                    style={{width: stepTabComponents.length * 250 }}
                >
                    {stepTabComponents}
                </AntdSteps>
            </div>
            {stepContentComponents[currentStepIndex]}
        </Card>
}