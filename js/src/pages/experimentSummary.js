import {Row, Col} from 'antd';
import * as React from "react";
import {ConfigurationCard, getConfigData, getStepComponent} from "../components/steps";

import {showNotification} from "../util";

export function ExperimentSummary ({experimentData, dispatch}) {

    const steps = experimentData.steps;
    if(steps.length < 1){
        showNotification('Step is empty');
        return ;
    }
    // steps length should > 0
    const groupSize = 2;
    const stepLen = steps.length;

    var buffer = [];
    const groups = [];
    for (let i = 0; i < stepLen; i++) {
        buffer.push(steps[i]);
        if(buffer.length === groupSize){
            groups.push([...buffer]);
            buffer = [];
        }else{
            if(i === stepLen - 1){
                groups.push([...buffer]);
                buffer = []
            }
        }
    }

    return <div>
            {
                groups.map((steps, index, arrar) => {
                    return <Row gutter={[4, 4]} key={index}>
                        {
                            steps.map((step, indexC, arrayC) => {
                                const cardTitle = step.displayName;
                                const stepType = step.type;
                                const Comp = getStepComponent(step.type);
                                let configurationData;
                                if(Comp !== undefined && Comp !== null){
                                    configurationData = Comp({stepData: step, dispatch: dispatch}).getDisplayConfigData()
                                }else{
                                    console.error("Internal error, unhandled step type: " + stepType);
                                    configurationData = {}
                                }
                                return <Col span={10} key={indexC} >
                                    <ConfigurationCard cardTitle={cardTitle} configurationData={getConfigData(configurationData, step.meta.configTip)}/>
                                </Col>
                            })
                        }
                    </Row>
                })
            }
        </div>
}
