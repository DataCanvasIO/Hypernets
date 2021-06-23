import {Card, Col, Form, Row, Switch, Table, Tabs, Tooltip, Select, Tag, Input} from "antd";
import React, { PureComponent } from 'react';
import { Empty } from 'antd';
import EchartsCore from './echartsCore';
import {showNotification} from "../util";
import {formatFloat, isEmpty} from "../util";
import {StepStatus, COMPONENT_SIZE, TABLE_ITEM_SIZE, Steps} from "../constants";

const { TabPane } = Tabs;
const CONFIGURATION_CARD_TITLE = "Configuration";

const COLUMN_ELLIPSIS = () => {
    return {
        ellipsis: {
            showTitle: false,

        },
        render: function (text, record, index) {
            return <Tooltip placement="topLeft" title={text}>
                {text}
            </Tooltip>
        }
}};


export class Line extends PureComponent {
    getOption = (min, max) => {
        return {
            parallelAxis: [],
            series: [{
                type: 'parallel',
                lineStyle: {
                    width: 2
                },
                data: [],
            }],
            visualMap: {
                min: min,
                max: max,
                left: 0,
                precision: 4,
                calculable: true,
                orient: 'vertical',
                bottom: '15%',
                hoverLink:true,
            },
        };
    };

    render() {
        const { style, className, data, xAxisProp, seriesData, loading, ...restProp } = this.props;
        if (!data || !data.hasOwnProperty('data')) {
            return (<Empty />);
        }
        const  rewards = [];
        for (var item of data.data){
            rewards.push(item.reward);
        }
        const options = this.getOption(Math.min(...rewards), Math.max(...rewards));
        data.param_names.forEach((name, index) => {
            options.parallelAxis.push({
                dim: index,
                name
            });
        });
        data.data.forEach((item, index) => {
            options['series']['data'] = new Array(data.param_names.length);
            options['series'][0]['data'][index] = item.params;
        });

        return (
            <EchartsCore
                // loadingOption={{ color: '#1976d2' }}
                option={ {...options} }
                // showLoading={loading}
                // style={style}
                // className={className}
            />
        );
    }
}

export function ConfigurationCard({configurationData, configurationTip = {}}) {

    const makeLabel = (label, maxLen=16) => {
        let displayLabel;
        if(label.length > maxLen){
            displayLabel = `${label.substring(0, maxLen - 3)}...`;
        }else{
            displayLabel = label
        }
        const tip = configurationTip[label];
        let tipContent;
        if(tip !== null && tip !== undefined){
            tipContent = `${label}\n${tip}`
        }else{
            tipContent = label
        }
        return <Tooltip title={tipContent}>
            {displayLabel}
        </Tooltip>
    };

    return <Form
        size={COMPONENT_SIZE}
        labelAlign={'right'}
        labelCol={{span: 8, offset: 0}}
        style={{align: 'right'}}
        layout="horizontal">
        {
            Object.keys(configurationData).sort( function (a, b) {
                return b.length - a.length
            }).map(key => {
                const v = configurationData[key];
                let content;
                if (v === undefined || v === null){
                    content = <Input value={"None"} disabled={true} />
                } else if((typeof  v)  === "boolean"){
                    content =  <Switch checked />
                } else  if(v instanceof Array){
                    content =   <Input value={  v.join(",") } disabled={true} />
                } else {
                    content =  <Input value={v} disabled={true}/>
                }
                return <Form.Item label={ makeLabel(key) } key={key}>
                    <span>{content}</span>
                </Form.Item>
            })
        }
    </Form>
}

export function DataCleaningStep({stepData}) {

    var removedFeaturesDataSource = null;
    if(stepData.status === StepStatus.Finish){
        const unselectedFeatures = stepData.extension.unselected_reason;
        if(!isEmpty(unselectedFeatures)){
            removedFeaturesDataSource = Object.keys(unselectedFeatures).map((value, index, arr) => {
                return {
                    key: index,
                    feature_name: value,
                    reason: unselectedFeatures[value]
                }
            });
        }
    }

    const removedFeaturesColumns = [
        {
            title: 'Feature name',
            dataIndex: 'feature_name',
            key: 'feature_name',
        },
        {
            title: 'Reason',
            dataIndex: 'reason',
            key: 'reason',
        }
    ];

    return <Row gutter={[2, 2]}>
        <Col span={10} >
            <Card title={CONFIGURATION_CARD_TITLE} bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                {
                    <ConfigurationCard configurationData={stepData.configuration.data_cleaner_args} configurationTip={{}}/>
                }
            </Card>
        </Col>

        <Col span={10} offset={2} >
            <Card title="Removed features" bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                <Table dataSource={removedFeaturesDataSource}
                       columns={removedFeaturesColumns}
                       size={COMPONENT_SIZE}
                       pagination={ { pageSize:  TABLE_ITEM_SIZE }}
                />
            </Card>
        </Col>
    </Row>

}

export function FeatureGenerationStep({stepData}){

    let dataSource;

    if(stepData.status === StepStatus.Finish){
        dataSource = stepData.extension.outputFeatures.map((value, index, arr) => {
            return {
                key: index,
                name: value.name,
                parentFeatures: value.parentFeatures !== null && value.parentFeatures.length > 0 ? value.parentFeatures.join(",") : "-",
                primitive: value.primitive
            }
        });
    }else{
        dataSource = null;
    }

    const columns = [
        {
            title: 'Feature',
            dataIndex: 'name',
            key: 'name',
            ...COLUMN_ELLIPSIS()
        }, {
            title: 'Sources',
            dataIndex: 'parentFeatures',
            key: 'parentFeatures',
            ...COLUMN_ELLIPSIS()
        },{
            title: 'Primitive',
            dataIndex: 'primitive',
            key: 'primitive',
            render: function (text, record, index) {
                if(text !== null && text !== undefined){
                    return <Tag color="#108ee9">{text}</Tag>;
                }else{
                    return '-';
                }
            }
        }
    ];

    return <Row gutter={[2, 2]}>
        <Col span={10} >
            <Card title={CONFIGURATION_CARD_TITLE} bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                {
                    <ConfigurationCard configurationData={stepData.configuration} configurationTip={Steps.FeatureGeneration.configTip}/>
                }
            </Card>
        </Col>

        <Col span={10} offset={2} >
            <Card title="Output features" bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                <Table dataSource={dataSource}
                       columns={columns}
                       pagination={ {defaultPageSize: TABLE_ITEM_SIZE, disabled: false, pageSize:  TABLE_ITEM_SIZE}}
                       size={COMPONENT_SIZE} />
            </Card>
        </Col>
    </Row>
}

export function CollinearityDetectionStep({stepData}){

    let dataSource;
    if(stepData.status === StepStatus.Finish){
        dataSource = stepData.extension.unselected_features?.map((value, index, arr) => {
            return {
                key: index,
                removed: value.removed,
                reserved: value.reserved,
            }
        });
    }else{
        dataSource = null;
    }

    const columns = [
        {
            title: 'Removed',
            dataIndex: 'removed',
            key: 'removed',
        },
        {
            title: 'Reserved',
            dataIndex: 'reserved',
            key: 'reserved',
        }
    ];

    return <Row gutter={[2, 2]}>
        <Col span={10} >
            <Card title={CONFIGURATION_CARD_TITLE} bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                {
                    <ConfigurationCard configurationData={stepData.configuration} configurationTip={Steps.CollinearityDetection.configTip}/>
                }
            </Card>
        </Col>

        <Col span={10} offset={2} >
            <Card title="Removed features" bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                <Table dataSource={dataSource}
                       columns={columns}
                       size={COMPONENT_SIZE}
                       pagination={ { pageSize:  TABLE_ITEM_SIZE }}
                />
            </Card>
        </Col>
    </Row>
}

export function DriftDetectionStep({stepData}){

    let driftFeatureAUCDataSource;
    if(stepData.status === StepStatus.Finish){
        driftFeatureAUCDataSource = stepData.extension.drifted_features_auc?.map((value, index, arr) => {
            return {
                key: index,
                feature: value.feature,
                score: formatFloat(value.score),
            }
        });
    }else{
        driftFeatureAUCDataSource = null;
    }

    const driftFeatureAUCColumns = [
        {
            title: 'Feature',
            dataIndex: 'feature',
            key: 'feature',
            ...COLUMN_ELLIPSIS()
        },
        {
            title: 'AUC',
            dataIndex: 'score',
            key: 'score',
        }
    ];



    const removedFeaturesInEpochColumns = [
        {
            title: 'Feature',
            dataIndex: 'feature',
            key: 'feature',
            ...COLUMN_ELLIPSIS()
        },
        {
            title: 'Importance',
            dataIndex: 'importance',
            key: 'importance',
        }
    ];

    return <><Row gutter={[2, 2]}>
        <Col span={10} >
            <Card title={CONFIGURATION_CARD_TITLE} bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                {
                    <ConfigurationCard configurationData={stepData.configuration} configurationTip={Steps.DriftDetection.configTip}/>
                }
            </Card>
        </Col>

        <Col span={10} offset={2} >
            <Card title="Drifted features AUC" bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                <Table dataSource={driftFeatureAUCDataSource}
                       columns={driftFeatureAUCColumns}
                       size={COMPONENT_SIZE}
                       pagination={ { pageSize:  7 }}
                />
            </Card>
        </Col>

    </Row>
        <Row gutter={[4, 4]}>
            <Col span={10} offset={12} >
                <Card title="Removed features in epochs" bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                    <Tabs defaultActiveKey="1" tabPosition={'top'} style={{ height: '100%', width: '100%'}}>
                        {
                            stepData.extension.removed_features_in_epochs?.map(epoch => {
                                return <TabPane tab={`Epoch ${epoch.epoch}`} key={epoch.epoch}>
                                    <Table dataSource={epoch.removed_features?.map((value, index, arr) => {
                                        return {
                                            key: index,
                                            feature: value.feature,
                                            importance: formatFloat(value.importance),
                                        }
                                    })} columns={removedFeaturesInEpochColumns}
                                           size={COMPONENT_SIZE}
                                           pagination={ { pageSize:  TABLE_ITEM_SIZE }}
                                    />
                                </TabPane>
                            })
                        }
                    </Tabs>
                </Card>
            </Col>
        </Row>
        </>
}

export function GeneralImportanceSelectionStep({stepData, configTip}){

    let dataSource;
    if(stepData.status === StepStatus.Finish){
        dataSource = stepData.extension.importances?.map((value, index, arr) => {
            return {
                key: index,
                featureName: value.name,
                importance: formatFloat(value.importance),
                dropped: value.dropped
            }
        });
    }else{
        dataSource = null;
    }

    const columns = [
        {
            title: 'Feature',
            dataIndex: 'featureName',
            key: 'featureName',
            ellipsis: {
                showTitle: false,

            },
            render: function (text, record, index) {
                return <Tooltip placement="topLeft" title={text}>
                    {text}
                </Tooltip>
            }
        }, {
            title: 'Importance',
            dataIndex: 'importance',
            key: 'importance',

        },{
            title: 'Status',
            dataIndex: 'dropped',
            key: 'dropped',
            render: function(text, record, index) {
                if(record.dropped){
                    return <Tag color="red">Unselected</Tag>
                }else{
                    return <Tag color="green">Selected</Tag>
                }
            }
        }
    ];

    return <Row gutter={[2, 2]}>
        <Col span={10} >
            <Card title={CONFIGURATION_CARD_TITLE} bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                {
                    <ConfigurationCard configurationData={stepData.configuration} configurationTip={configTip}/>
                }
            </Card>
        </Col>

        <Col span={10} offset={2} >
            <Card title="Importances" bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                <Table dataSource={dataSource}
                       columns={columns}
                       pagination={ {defaultPageSize: TABLE_ITEM_SIZE, disabled: false, pageSize:  TABLE_ITEM_SIZE}}
                       size={COMPONENT_SIZE} />
            </Card>
        </Col>
    </Row>
}

export function PseudoLabelStep({stepData, dispatch}) {

    const getProbaDensityEchartOpts = (labelName) => {
        const seriesData = [];

        if(!isEmpty(labelName)){
            const probabilityDensityLabelData = probabilityDensity[labelName];
            const gaussianData = probabilityDensityLabelData['gaussian'];
            for(var i = 0 ;i < gaussianData['X'].length; i++){
                seriesData.push([gaussianData['X'][i], gaussianData['probaDensity'][i]])
            }
        }
        return {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                }
            },
            xAxis: {
                type: 'category',
                boundaryGap: false
            },
            yAxis: {
                type: 'value',
                boundaryGap: [0, '30%']
            },
            series: [
                {
                    type: 'line',
                    smooth: 0.6,
                    symbol: 'none',
                    lineStyle: {
                        color: '#5470C6',
                        width: 3
                    },
                    areaStyle: {},
                    data: seriesData
                }
            ]
        };
    };

    const probabilityDensity = stepData.extension.probabilityDensity;
    const labels =  isEmpty(probabilityDensity) ? null : Object.keys(probabilityDensity);
    let selectedLabel;
    if(stepData.status === StepStatus.Finish){
        if(labels === null || labels.length < 2){
            showNotification("Pseudo step is success but labels is empty");
            return ;
        }
        selectedLabel = stepData.extension.selectedLabel;
    }else{
        selectedLabel = null;
    }

    const probaDensityChartOption = getProbaDensityEchartOpts(selectedLabel);

    const onLabelChanged = (value) => {
        dispatch(
            {
                type: 'probaDensityLabelChange',
                payload: {
                    stepIndex: stepData.index,
                    selectedLabel: value
                }
            }
        )
    };

    if (!isEmpty(labels)){
        if(labels.length < 2){
            console.error("labels should not < 2");
        }
    }

    const defaultLabel = isEmpty(labels) ? "None" : labels[0];

    const samplesObj = stepData.extension.samples;

    const samplesDataSource = isEmpty(samplesObj) ? null : Object.keys(samplesObj).map((value,index, array) => {
        return {
            key: index,
            label: value,
            count: samplesObj[value]
        }
    });


    const samplesColumns = [
        {
            title: 'Label',
            dataIndex: 'label',
            key: 'Label',
        },
        {
            title: 'Count',
            dataIndex: 'count',
            key: 'Count',
        }
    ];



    const { Option } = Select;

    return <>
        <Row gutter={[2, 2]}>
            <Col span={10} >
                <Card title={CONFIGURATION_CARD_TITLE} bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                    {
                        <ConfigurationCard configurationData={stepData.configuration} configurationTip={Steps.PsudoLabeling.configTip}/>
                    }
                </Card>
            </Col>

            <Col span={10} offset={2} >
                <Card title="Density Plot of Probability" bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                        <span>
                             <span style={{marginLeft: '20px'}}>Select label: </span>
                             <Select defaultValue={ selectedLabel } value={selectedLabel} style={{ width: 280 }} onChange={onLabelChanged} disabled={ isEmpty(labels)} >
                                {
                                    isEmpty(labels) ? null: labels.map( v => {
                                        return <Option value={v}>{v}</Option>
                                    })
                                }
                            </Select>
                        </span>
                    <EchartsCore option={probaDensityChartOption}/>
                </Card>
            </Col>
        </Row>
        <Row gutter={[4, 4]}>
            <Col span={10} offset={12} >
                <Card title="Number of samples" bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE} >
                    <Table dataSource={samplesDataSource}
                        columns = {samplesColumns}
                        pagination={false}
                        size={COMPONENT_SIZE} />
                </Card>
            </Col>
        </Row>
    </>
}

export function EnsembleStep({stepData}) {
    let scores;
    let weights;
    if(stepData.status === StepStatus.Finish){
        scores = stepData.extension.scores;
        scores = scores === null ? [] : scores;

        weights = stepData.extension.weights;
        weights = weights === null ? [] : weights;
    }else{
        scores =  null;
        weights = null;
    }

    const getLiftEchartOpts = () => {
        const yLabels = scores !== null && scores !== undefined ?  Array.from({length:scores.length}, (v,k) => k) : [];
        return  {
            xAxis: {
                type: 'category',
                data: yLabels
            },
            yAxis: {
                type: 'value'
            },
            series: [{
                data: scores,
                type: 'line',
                smooth: true
            }]
        };
    };

    const getWeightsEchartOpts = () => {
        const yLabels = weights !== null && weights !== undefined ?  Array.from({length:weights.length}, (v,k) => k) : [];
        return {
                tooltip: {
                    trigger: 'axis',
                        axisPointer: {
                        type: 'shadow'
                    }
                },
                legend: {
                    data: []
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '3%',
                    containLabel: true
                },
                xAxis: {
                    type: 'value',
                    boundaryGap: [0, 0.01]
                },
                yAxis: {
                    type: 'category',
                    data: [...yLabels].reverse(),
                    min: 0,
                    max: 1
                },
                series: [
                    {
                        name: 'weight',
                        type: 'bar',
                        data: [...weights].reverse()
                    }
                ]
            }
        };


    return <>
        <Row gutter={[2, 2]}>
        <Col span={10} >
            <Card title={CONFIGURATION_CARD_TITLE} bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                {
                    <ConfigurationCard configurationData={stepData.configuration} configurationTip={Steps.Ensemble.configTip}/>
                }
            </Card>
        </Col>

        <Col span={10} offset={2} >
            <Card title="Weight" bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>

                <EchartsCore option={getWeightsEchartOpts()}/>
            </Card>
        </Col>
        </Row>

        <Row gutter={[2, 2]}>
            <Col span={10} offset={12} >
                <Card title="Lifting" bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                    <EchartsCore option={getLiftEchartOpts()}/>
                </Card>
            </Col>
        </Row>
    </>
}

export function FinalTrainStep({stepData}) {

    return <>
        <Row gutter={[4, 4]}>
            <Col span={10} >
                <Card title={CONFIGURATION_CARD_TITLE} bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                    {
                        <ConfigurationCard configurationData={stepData.configuration} configurationTip={Steps.FinalTrain.configTip}/>
                    }
                </Card>
            </Col>

            <Col span={10} offset={2} >
                <Card title="Estimator" bordered={false} style={{ width: '100%' }} size={COMPONENT_SIZE}>
                    {
                        <ConfigurationCard configurationData={stepData.extension}/>
                    }
                </Card>
            </Col>
        </Row>
    </>
}
