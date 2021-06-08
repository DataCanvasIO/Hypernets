import React from 'react';
import  { useState, useEffect } from  "react";

import "antd/dist/antd.css";
import { Select, Button, Switch, Table, Steps, Progress, Card, Slider,  Form, Radio, Row, Col, Tooltip} from 'antd';
import "antd/dist/antd.css";
import { notification } from 'antd';
import {connect, Provider} from "react-redux";

import { Scrollbars } from 'react-custom-scrollbars';
import {CollinearityDetectionStep, DriftDetectionStep, EnsembleStep, FeatureSelectionStep, PseudoLabelStep} from '../components/steps'
import { StepsKey } from '../constants'
import useBreakpoint from "antd/es/grid/hooks/useBreakpoint";
import { DataCleaningStep} from '../components/dataCleaningStep'
import { PipelineOptimizationStep} from '../components/pipelineSearchStep'
const { Step } = Steps;

export const showNotification = (message) => {
    notification.error({
        key: '123',
        message: message,
        duration: 10,
    });
};

// step status: wait, process, finish, error




const StepUIStatus = {
    Wait : 'wait',
    Process : 'process',
    Finish : 'finish',
    Fail : 'fail'
};

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


    // if(newStepData !== null && newStepData !== undefined){
    //     experimentData.steps[newStepData.index] = newStepData;
    //     // update next step running status
    //     if(newStepData.status === StepUIStatus.Finish){
    //         if(newStepData.index < experimentData.steps.length - 1){  // Current update step is not the latest
    //             experimentData.steps[newStepData.index+1].status = StepUIStatus.Process
    //         }
    //     }
    // }
    const getProcessBarStatus  = () => {
        var processFinish = true;
        for (var step of experimentData.steps) {
            if (step.status === StepUIStatus.Fail) {
                return ProgressBarStatus.Exception
            }
            if(step.status !== StepUIStatus.Finish){  // all step is finish so the ProcessBar is succeed
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
                <DataCleaningStep data={stepData}/>
            );
        }else if(stepType  === StepsKey.CollinearityDetection.type){
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
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.SpaceSearch.name} key={`step_${stepData.name}`} />
            );
            stepContentComponents.push(
                <PipelineOptimizationStep stepData={stepData} />
            );

        } else if(stepType  === StepsKey.FeatureSelection.type){
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.FeatureSelection.name} key={`step_${stepData.name}`} />
            );
            stepContentComponents.push(
                <FeatureSelectionStep stepData={stepData} />
            );

        } else if(stepType  === StepsKey.PsudoLabeling.type){
            // todo add psudo
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.PsudoLabeling.name} key={`step_${stepData.name}`} />
            );
            stepContentComponents.push(
                <PseudoLabelStep stepData={stepData} dispatch={dispatch}/>
            );

        } else if(stepType  === StepsKey.ReSpaceSearch.type){
            // todo add ReSpaceSearch
            stepTabComponents.push(
                <Step status={stepData.status} title={StepsKey.ReSpaceSearch.name} key={`step_${stepData.name}`} />
            );
            stepContentComponents.push(
                <PipelineOptimizationStep stepData={stepData} />
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


    // useEffect(() => {
    //     if(cleanStepData !== null && cleanStepData !== undefined){
    //         setCleanDataState(cleanStepData);
    //     }
    //     // Redux should check configData is the same as last
    // }, [cleanStepData]);

    const onStepChange = (c) => {
        setCurrentStepIndex(c);
    };

    // navigation, default
    // const processPercentage = getProcessPercentage()
    //                   fixme  current={currentStepIndex}
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