import {Row, Col, Divider, Card} from 'antd';
import * as React from "react";
import {ConfigurationCard} from "../components/steps";

import {StepsKey, getStepName} from '../constants';
import {showNotification} from "../util";

export function ExperimentSummary ({experimentData, dispatch}) {

    const steps = experimentData.steps;
    if(steps.length < 1){
        showNotification('Step is empty');
        return ;
    }
    // steps length should > 1
    const groupSize = 2;
    const stepLen = steps.length;

    var buffer = [];
    const groups = [];
    for (let i = 0; i < stepLen; i++) {
        var c = {...steps[i].configuration};
        if(steps[i].type === StepsKey.DataCleaning.type){
            c = c.data_cleaner_params
        } else if (steps[i].type === StepsKey.SpaceSearch.type ){
            c.earlyStopping = null
        }
        c.StepName = getStepName(steps[i].type);
        buffer.push(c);
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
                groups.map((configs, index, arrar) => {
                    return <Row gutter={[4, 4]} key={index}>
                        {
                            configs.map((valueC, indexC, arrayC) => {
                                const stepName = valueC['StepName'];
                                delete valueC['StepName'];
                                return <Col span={10} key={indexC} >
                                    <Card title={stepName} bordered={false} style={{ width: '100%' }}>
                                        <ConfigurationCard configurationData={valueC}/>
                                    </Card>
                                </Col>
                            })
                        }
                    </Row>
                })
            }
        </div>
}
