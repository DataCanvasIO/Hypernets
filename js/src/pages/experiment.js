import React from 'react';
import  { useState, useEffect } from  "react";

import "antd/dist/antd.css";
import { Select, Button, Switch, Table, Steps, Progress, Card, Slider,  Form, Radio, Row, Col, Tooltip} from 'antd';
import "antd/dist/antd.css";
import { notification } from 'antd';
import {connect, Provider} from "react-redux";

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
import {StepsKey, StepStatus} from '../constants'
import useBreakpoint from "antd/es/grid/hooks/useBreakpoint";
import { PipelineOptimizationStep} from '../components/pipelineSearchStep'
const { Step } = Steps;


const ProgressBarStatus = {
    Success: 'success',
    Exception : 'exception',
    Normal : 'normal',
    Active : 'active',
};

export function ExperimentUI ({experimentData, dispatch} ) {

    const [currentStepIndex , setCurrentStepIndex] = useState(0);
    const [stepTabs, setStepTabs] = useState([]);
    const [stepContents, setStepContents] = useState([]);

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

    experimentData.steps.forEach(stepData=> {
        const stepType = stepData.type;
        if(stepType  === StepsKey.DataCleaning.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.DataCleaning.name} key={stepData.name}/>
            );
            stepContentComponents.push(
                <DataCleaningStep stepData={stepData}/>
            );
        }else if(stepType  === StepsKey.FeatureGeneration.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.FeatureGeneration.name} key={stepData.name} />
            );
            stepContentComponents.push(
                <FeatureGenerationStep stepData={stepData}/>
            );
        } else if(stepType  === StepsKey.CollinearityDetection.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.CollinearityDetection.name} key={stepData.name} />
            );
            stepContentComponents.push(
                <CollinearityDetectionStep stepData={stepData}/>
            );
        } else if(stepType  === StepsKey.DriftDetection.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.DriftDetection.name} key={stepData.name} />
            );
            stepContentComponents.push(
                <DriftDetectionStep stepData={stepData}/>
            );
        } else if(stepType  === StepsKey.SpaceSearch.type){
            const stepName = stepData.configuration.name;
            let title;
            if (stepName === StepsKey.TwoStageSpaceSearch.key){
                title = StepsKey.TwoStageSpaceSearch.name;
            }else{
                title = StepsKey.SpaceSearch.name;
            }
            stepTabComponents.push(
                <Step status={stepData.status} title={title} key={`step_${stepName}`} />
            );
            stepContentComponents.push(
                <PipelineOptimizationStep stepData={stepData} key={`step_ui_${stepName}`} />
            );
        } else if(stepType  === StepsKey.FeatureSelection.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.FeatureSelection.name} key={`step_${stepData.name}`} />
            );
            stepContentComponents.push(
                <GeneralImportanceSelectionStep stepData={stepData} key={'FeatureSelection'} />
            );
        }  else if(stepType  === StepsKey.PermutationImportanceSelection.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.PermutationImportanceSelection.name} key={`step_${stepData.name}`} />
            );
            stepContentComponents.push(
                <GeneralImportanceSelectionStep stepData={stepData} key={'PermutationImportanceSelection'} />
            );
        } else if(stepType  === StepsKey.PsudoLabeling.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.PsudoLabeling.name} key={`step_${stepData.name}`} />
            );
            stepContentComponents.push(
                <PseudoLabelStep stepData={stepData} dispatch={dispatch}/>
            );

        } else if(stepType  === StepsKey.FinalTrain.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.FinalTrain.name} key={stepData.name} />
            );
            stepContentComponents.push(
                <FinalTrainStep stepData={stepData}/>
            );
        } else if(stepType  === StepsKey.Ensemble.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.Ensemble.name} key={stepData.name} />
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
            <div style={ {width: '100%', overflow: 'auto', marginTop: 20} }>
                <Steps
                    type="navigation"
                    size="small"
                    current={currentStepIndex}
                    onChange={onStepChange}
                    className="site-navigation-steps"
                >
                    {stepTabComponents}
                </Steps>
            </div>
            {stepContentComponents[currentStepIndex]}
        </Card>
}