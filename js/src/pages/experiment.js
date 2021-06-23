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

import {Steps, StepStatus} from '../constants'
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


    const getProcessBarStatus  = () => {
        var processFinish = true;
        for (var step of experimentData.steps) {
            if (step.status === StepStatus.Error) {
                return ProgressBarStatus.Exception
            }
            if(step.status !== StepStatus.Finish){  // all step is finish so the ProcessBar is succeed
                processFinish = false;
            }
        }
        if(processFinish){
            return ProgressBarStatus.Success
        }else{
            return ProgressBarStatus.Active
        }
    };

    const getProcessPercentage = () => {
        // 1. find last finished step index
        var lastFinishedStepIndex = -1;
        for(var i = experimentData.steps.length - 1; i >= 0 ; i--){
            if(experimentData.steps[i].status === 'finish'){
                lastFinishedStepIndex = i;
                break;
            }
        }
        // 2. last finished step index / total step
        return (((lastFinishedStepIndex + 1) / experimentData.steps.length) * 100).toFixed(0);
    };
    const stepStyle = {marginLeft: 20, marginRight: 20};

    experimentData.steps.forEach(stepData=> {
        const stepType = stepData.type;
        if(stepType  === Steps.DataCleaning.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={Steps.DataCleaning.name} key={stepData.name} style={stepStyle} />
            );
            stepContentComponents.push(
                <DataCleaningStep stepData={stepData}/>
            );
        }else if(stepType  === Steps.FeatureGeneration.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={Steps.FeatureGeneration.name} key={stepData.name} style={stepStyle} />
            );
            stepContentComponents.push(
                <FeatureGenerationStep stepData={stepData}/>
            );
        } else if(stepType  === Steps.CollinearityDetection.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={Steps.CollinearityDetection.name} key={stepData.name} style={stepStyle} />
            );
            stepContentComponents.push(
                <CollinearityDetectionStep stepData={stepData}/>
            );
        } else if(stepType  === Steps.DriftDetection.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={Steps.DriftDetection.name} key={stepData.name} style={stepStyle} />
            );
            stepContentComponents.push(
                <DriftDetectionStep stepData={stepData}/>
            );
        } else if(stepType  === Steps.SpaceSearch.type){
            const stepName = stepData.configuration.name;
            let title;
            if (stepName === Steps.TwoStageSpaceSearch.key){
                title = Steps.TwoStageSpaceSearch.name;
            }else{
                title = Steps.SpaceSearch.name;
            }
            stepTabComponents.push(
                <Step status={stepData.status} title={title} key={`step_${stepName}`} style={stepStyle} />
            );
            stepContentComponents.push(
                <PipelineOptimizationStep stepData={stepData} key={`step_ui_${stepName}`} />
            );
        } else if(stepType  === Steps.FeatureSelection.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={Steps.FeatureSelection.name} key={`step_${stepData.name}`} style={stepStyle} />
            );
            stepContentComponents.push(
                <GeneralImportanceSelectionStep stepData={stepData} configTip={Steps.FeatureSelection.configTip} key={'FeatureSelection'} />
            );
        }  else if(stepType  === Steps.PermutationImportanceSelection.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={Steps.PermutationImportanceSelection.name} key={`step_${stepData.name}`} style={stepStyle} />
            );
            stepContentComponents.push(
                <GeneralImportanceSelectionStep stepData={stepData} configTip={Steps.PermutationImportanceSelection.configTip} key={'PermutationImportanceSelection'} />
            );
        } else if(stepType  === Steps.PsudoLabeling.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={Steps.PsudoLabeling.name} key={`step_${stepData.name}`} style={stepStyle} />
            );
            stepContentComponents.push(
                <PseudoLabelStep stepData={stepData} dispatch={dispatch}/>
            );

        } else if(stepType  === Steps.FinalTrain.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={Steps.FinalTrain.name} key={stepData.name} style={stepStyle} />
            );
            stepContentComponents.push(
                <FinalTrainStep stepData={stepData}/>
            );
        } else if(stepType  === Steps.Ensemble.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={Steps.Ensemble.name} key={stepData.name} style={stepStyle} />
            );
            stepContentComponents.push(
                <EnsembleStep stepData={stepData}/>
            );
        }
        else {
            // showNotification("Unknown step type");
        }
    });

    const onStepChange = (c) => {
        setCurrentStepIndex(c);
    };

    return <Card title="Experiment progress" bordered={false} style={{ width: '100%' }}>
            <Progress percent={  getProcessPercentage() } status={ getProcessBarStatus ()} />
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