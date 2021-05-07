import React from 'react';
import  { useState, useEffect } from  "react";

import "antd/dist/antd.css";
import { Select, Button, Switch, Table, Steps, Progress, Card, Slider,  Form, Radio, Row, Col, Tooltip} from 'antd';
import "antd/dist/antd.css";
import { notification } from 'antd';

import { Scrollbars } from 'react-custom-scrollbars';
import { DataCleaningStep, CollinearityDetectionStep, DriftDetectionStep } from '../components/steps'
import { StepsKey } from '../constants'
import useBreakpoint from "antd/es/grid/hooks/useBreakpoint";

const { Step } = Steps;

export const showNotification = (message) => {
    notification.error({
        key: '123',
        message: message,
        duration: 10,
    });
};

export function ExperimentUI ({configData}) {

    const [currentStepIndex , setCurrentStepIndex] = useState(0);
    const [experimentStepsComponent, setExperimentStepsComponent] = useState([]);
    const [stepsTab, setStepsTab] = useState([]);

    useEffect(() => {
        renderExperiments(configData);
    }, [configData]);

    const onStepChange = (c) => {
        setCurrentStepIndex(c);
    };

    function renderExperiments(experimentConfigData) {
        // const experimentStepsData = experimentConfigData.steps;
        const _stepTab = [];
        const _experimentStepsComponent = [];

        experimentConfigData.steps.forEach( stepData=> {
            if(stepData.kind  === StepsKey.DataCleaning.kind){
                _stepTab.push(
                    <Step status="finish" title={StepsKey.DataCleaning.name} />
                );
                _experimentStepsComponent.push(
                    <DataCleaningStep stepData={stepData}/>
                );
            } else if(stepData.kind  === StepsKey.CollinearityDetection.kind){
                _stepTab.push(
                    <Step status="finish" title={StepsKey.CollinearityDetection.name} />
                );
                _experimentStepsComponent.push(
                    <CollinearityDetectionStep stepData={stepData}/>
                );
            } else if(stepData.kind  === StepsKey.DriftDetection.kind){
                _stepTab.push(
                    <Step status="finish" title={StepsKey.DriftDetection.name} />
                );
                _experimentStepsComponent.push(
                    <DriftDetectionStep stepData={stepData}/>
                );
            }
            else{
                showNotification("Unknown error");
            }
        });

        setStepsTab(_stepTab);
        setExperimentStepsComponent(_experimentStepsComponent);
    }


    // navigation, default
    return <Card title="Experiment progress" bordered={false} style={{ width: '100%' }}>
            <Progress percent={50} status="active" />
                <div style={ {width: '100%', overflow: 'auto', marginTop: 20} }>
                    <Steps
                        type="navigation"
                        size="small"
                        current={currentStepIndex}
                        onChange={onStepChange}
                        className="site-navigation-steps"
                    >
                        {
                        /*
                        *  <Tooltip title="prompt text"> </Tooltip>
                        */
                            // antd now not supported Tooltip with Step: https://github.com/ant-design/ant-design/issues/20957
                        /*
                        * <Step status="finish" title="Data cleaning" />
                        <Step status="finish" title="Collinearity detection" />
                        <Step status="process" title="Drift detection" />
                        <Step status="process" title="Pipeline optimization" />
                        <Step status="process" title="Feature selection" />
                        <Step status="process" title="Psudo labeling" />
                        <Step status="process" title="Pipeline re-optimization" />
                        <Step status="wait" title="Ensemble" />
                        * */
                        }

                        {stepsTab}


                    </Steps>

                </div>

            {experimentStepsComponent[currentStepIndex]}

        </Card>
}