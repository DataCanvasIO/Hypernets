import PropTypes from "prop-types";

import React, {PureComponent, useEffect, useState} from 'react';

import {Card, Col, Form, Row, Switch, Table, Tabs, Tooltip, Select, Tag, Input, Result, Progress} from "antd";

import * as echarts from "echarts/lib/echarts";
import 'echarts/lib/chart/heatmap';
import 'echarts/lib/component/legend';
import 'echarts/lib/component/dataZoom';

import { clear as throttleClear } from 'echarts/lib/util/throttle';
import { BarChart, LineChart, ScatterChart } from 'echarts/charts';
import { GridComponent, MarkPointComponent, TooltipComponent, ToolboxComponent, LegendComponent, TitleComponent } from 'echarts/components';

import EchartsCore from './echartsCore';

import {formatHumanDate, showNotification, formatFloat, isEmpty, notEmpty, getOrDefault} from "../util";
import {StepStatus, COMPONENT_SIZE, TABLE_ITEM_SIZE, ActionType, TWO_STAGE_SUFFIX} from "../constants";


const {TabPane} = Tabs;
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
    }
};

const makeColumn  = (title, dataIndex, key=null, withEllipsis=true)=> {
    let columnKey;
    if(key === null){
        columnKey = dataIndex
    }else{
        columnKey = key
    }
    var ellipsisConfig = {};
    if(withEllipsis){
        ellipsisConfig = COLUMN_ELLIPSIS()
    }
    return {
        title: title,
        dataIndex: dataIndex,
        key: columnKey,
        ...ellipsisConfig
    };
};

export function getConfigData(configs, tips) {

    if(isEmpty(configs)){
        configs = {}
    }
    if(isEmpty(tips)){
        tips = {}
    }

    return Object.keys(configs).map(key => {
        const v = configs[key];
        const tip = tips[key];
        return {
            name: key,
            value: v,
            tip: tip
        }
    });
}

export function FeatureNumCard({stepData, hasOutput = true}) {  // why not use `Collapse` ? it has different style with `Card`
    let featuresData;
    const validate = (keys, featuresData) => {
        keys.forEach(key => {
                const value = featuresData[key];
                if (value === undefined || value === null) {
                    console.error(`features.${key} is null, step data is:`);
                    console.error(stepData);
                }
            }
        )
    };

    if (stepData.status === StepStatus.Finish) {
        featuresData = stepData.extension.features;
        if (featuresData !== null && featuresData !== undefined) {
            if(hasOutput){
                validate(['inputs', 'outputs', 'increased', 'reduced'], featuresData)
            }else{
                validate(['inputs'], featuresData)  // if no outputs, only display inputs
            }

        } else {
            console.error(`step index = ${stepData.index} is finish but features data is null, step data is:`);
            console.error(stepData);
        }
    } else {
        featuresData = {
            inputs: null,
            outputs: null,
            increased: null,
            reduced: null
        }
    }

    const _render = (features) => {
        if (features !== undefined && features !== null) {
            return <Tooltip title={features.join(',')}>
                {features.length}
            </Tooltip>
        } else {
            return <Tooltip title={"This field is null"}>
                None
            </Tooltip>
        }
    };

    const formData = [
        {
            name: "Inputs ",
            tip: null,
            value: featuresData.inputs,
            render: _render
        }
    ];
    if (hasOutput) {
        const items = [
            {
                name: "Outputs",
                tip: null,
                value: featuresData.outputs,
                render: _render
            }, {
                name: "Increased",
                tip: null,
                value: featuresData.increased,
                render: _render
            }, {
                name: "Reduced",
                tip: null,
                value: featuresData.reduced,
                render: _render
            }
        ];
        for (var item of items) {
            formData.push(item)
        }
    }
    return <Card title="Features" bordered={false} style={{width: '100%'}} size={'small'}>
        <ConfigurationForm
            configurationData={formData}
            sort={false}
        />
    </Card>
}

export function StepStatusCard({stepData}) {
    const status = stepData.status;
    const start_datetime = stepData.start_datetime;
    const end_datetime = stepData.end_datetime;

    const fillZero = (num) => {
        if (num < 10) {
            return `0${num}`
        } else {
            return `${num}`
        }
    };

    const formatTimestamp = (timestamp) => {
        if (timestamp !== undefined && timestamp !== null) {
            const d = new Date(timestamp * 1000);
            return `${d.getFullYear()}-${fillZero(d.getMonth() + 1)}-${fillZero(d.getDate())} ${fillZero(d.getHours())}:${fillZero(d.getMinutes())}:${fillZero(d.getSeconds())}`
        } else {
            return '-'
        }
    };
    var elapsed = '-';
    if (status === StepStatus.Finish) {  // calc elapsed time
        if (start_datetime !== undefined && start_datetime !== null) {
            if (end_datetime !== undefined && end_datetime !== null) {
                elapsed = formatHumanDate(end_datetime - start_datetime);
            } else {
                console.error(`status is ${status} but end_datetime is null `);
            }
        } else {
            console.error(`status is ${status} but start_datetime is null `);
        }
    }

    const basicRender = (v) => {
        return <span>
            {v}
        </span>
    };

    const configDict = [
        {
            name: "Start datetime",
            tip: null,
            value: formatTimestamp(start_datetime),
            render: basicRender
        },
        {
            name: "End datetime",
            tip: null,
            value: formatTimestamp(end_datetime),
            render: basicRender
        },
        {
            name: "Elapsed",
            tip: null,
            value: elapsed,
            render: basicRender
        },
        {
            name: "Status",
            tip: null,
            value: stepData.status,
            render: function (v) {
                let color;
                if (v === StepStatus.Wait) {
                    color = 'gray'
                } else if (v === StepStatus.Process) {
                    color = '#2db7f5'
                } else if (v === StepStatus.Finish) {
                    color = '#87d068'
                } else if (v === StepStatus.Skip) {
                    color = 'orange'
                } else if (v === StepStatus.Error) {
                    color = '#f50'
                } else {
                    color = 'black'
                }
                return <Tag style={{color: color}}>
                    {v}
                </Tag>
            }
        }
    ];

    return <Card title="Status" bordered={false} style={{width: '100%'}} size={'small'}>
        <ConfigurationForm
            configurationData={configDict}
            sort={false}
        />
    </Card>
}

/***
 *  {
        name: "",
        tip: "",
        value: "",
        render: function (v) {

        }
    }
 * @param configurationData
 * @param sort
 * @returns {*}
 * @constructor
 */
export function ConfigurationForm({configurationData, sort = true}) {

    const makeLabel = (label, tip, maxLen = 30) => {
        let displayLabel;
        if (label.length > maxLen) {
            displayLabel = `${label.substring(0, maxLen - 3)}...`;
        } else {
            displayLabel = label
        }
        let tipContent;
        if (tip !== null && tip !== undefined) {
            tipContent = `${label}\n${tip}`
        } else {
            tipContent = label
        }
        return <Tooltip title={tipContent}>
            {displayLabel}
        </Tooltip>
    };

    if (sort) {
        configurationData = configurationData.sort(function (a, b) {
            return b.name.length - a.name.length
        })
    }

    const defaultRender = (v) => {
        if (v === undefined || v === null) {
            return <Input value={"None"} disabled={true}/>
        } else if ((typeof v) === "boolean") {
            return <Switch checked={v} disabled/>
        } else if (v instanceof Array) {
            return <Input value={v.join(",")} disabled={true}/>
        } else {
            return <Input value={v} disabled={true}/>
        }
    };

    const renderItem = (config) => {
        const configRender = config.render;
        const v = config.value;
        if (config.render !== undefined && configRender !== null) {
            return configRender(v);
        } else {
            return defaultRender(v)
        }
    };

    return <Form
        size={COMPONENT_SIZE}
        labelAlign={'right'}
        labelCol={{span: 12, offset: 0}}
        style={{align: 'right'}}
        layout="horizontal">
        {
            configurationData.map(config => {
                return <Form.Item style={{marginBottom: 12}} label={makeLabel(config.name)} key={config.name}>
                    {
                        renderItem(config)
                    }
                </Form.Item>
            })
        }
    </Form>
}

export function ConfigurationCard({configurationData, cardTitle = CONFIGURATION_CARD_TITLE, sort = true}) {
    return <Card title={cardTitle} bordered={false} style={{width: '100%'}} size={COMPONENT_SIZE}>
        {
            <ConfigurationForm configurationData={configurationData} sort={sort}/>
        }
    </Card>
}

export function SkippedStepContent({style = {marginTop: 20, marginLeft: 20}}) {

    const subTitle = `Step "${PseudoLabelStep.getTypeName()}" and step "${PermutationImportanceSelectionStep.getTypeName()}" have no impact on the dataset.` // fixme: distribute mode has another name
    return <Result
        style={{...style}}
        status="info"
        title="This step was  skipped."
        subTitle={subTitle}
    />
}


export class BaseStepComponent extends React.Component{

    static getDisplayName() { return null;}
    static getTypeName() { return  null; }
    static getConfigTip() { return  {}; }
    getConfigTip() { return  BaseStepComponent.getConfigTip(); }  // instance can not call static method, so copy to instance

    constructor(props) {
        super(props);
        this.stepData = props.stepData;
    }

    getDisplayConfigData(){
        return getConfigData(this.stepData.configuration, this.getConfigTip());
    }

}

export class StepWithStdConfigAndStatusAndFeatureCard extends BaseStepComponent{

    getBasicCards(hasOutput = true){
        const stepData = this.stepData;
        return <Col span={11}>
            <Row gutter={[2, 2]}>
                <ConfigurationCard configurationData={this.getDisplayConfigData()}/>
            </Row>
            <Row gutter={[2, 2]}>
                <StepStatusCard stepData={stepData}/>
            </Row>
            <Row gutter={[2, 2]}>
                <FeatureNumCard stepData={stepData} hasOutput={hasOutput}/>
            </Row>
        </Col>
    }

}

export class DataCleaningStep extends StepWithStdConfigAndStatusAndFeatureCard {

    static getDisplayName() { return "Data cleaning";}
    static getTypeName() { return  "DataCleanStep"; }

    getDisplayConfigData(){
        return getConfigData(this.stepData.configuration.data_cleaner_args, this.getConfigTip());
    }

    render() {
        const stepData = this.props.stepData;
        var removedFeaturesDataSource = null;
        if (stepData.status === StepStatus.Finish) {
            const unselectedFeatures = stepData.extension.unselected_reason;
            if (!isEmpty(unselectedFeatures)) {
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
            makeColumn('Feature name', 'feature_name'),
            makeColumn('Reason', 'reason')
        ];

        return <Row gutter={[2, 2]}>
            {this.getBasicCards()}
            <Col span={11} offset={2}>
                <Card title="Removed features" bordered={false} style={{width: '100%'}} size={COMPONENT_SIZE}>
                    <Table dataSource={removedFeaturesDataSource}
                           columns={removedFeaturesColumns}
                           size={COMPONENT_SIZE}
                           pagination={{pageSize: TABLE_ITEM_SIZE, hideOnSinglePage: true}}
                    />
                </Card>
            </Col>
        </Row>
    }
}

export class FeatureGenerationStep extends StepWithStdConfigAndStatusAndFeatureCard{

    static getDisplayName() { return "Feature generation";}
    static getTypeName() { return  "FeatureGenerationStep"; }
    static getConfigTip() {
        return {
            strategy:  "Strategy to select features",
            threshold:  "Confidence threshold of feature_importance. Only valid when *feature_selection_strategy* is 'threshold'.",
            quantile:  "Confidence quantile of feature_importance. Only valid when *feature_selection_strategy* is 'quantile'.",
            number: "Expected feature number to keep. Only valid when *feature_selection_strategy* is 'number'.",
        }
    }
    getConfigTip() { return  FeatureGenerationStep.getConfigTip(); }

    constructor(props) {
        super(props);
        this.state = {
            featurePageSize: TABLE_ITEM_SIZE * 2
        }
    }

    getFeaturePageSize() {
        return this.state.featurePageSize
    };

    setFeaturePageSize(featurePageSize){
        this.setState({featurePageSize: featurePageSize})
    }

    render() {
        const stepData = this.stepData;
        let dataSource;
        if (stepData.status === StepStatus.Finish) {
            dataSource = stepData.extension.outputFeatures.map((value, index, arr) => {
                return {
                    key: index,
                    name: value.name,
                    parentFeatures: value.parentFeatures !== null && value.parentFeatures.length > 0 ? value.parentFeatures.join(",") : "-",
                    primitive: value.primitive
                }
            });
        } else {
            dataSource = null;
        }

        const columns = [
            makeColumn('Feature', 'name'),
            makeColumn('Sources', 'parentFeatures'),
            {
                title: 'Primitive',
                dataIndex: 'primitive',
                key: 'primitive',
                render: function (text, record, index) {
                    if (text !== null && text !== undefined) {
                        return <Tag color="#108ee9">{text}</Tag>;
                    } else {
                        return '-';
                    }
                }
            }
        ];

        return <Row gutter={[2, 2]}>
            {this.getBasicCards()}
            <Col span={11} offset={2}>
                <Card title="Output features" bordered={false} style={{width: '100%'}} size={COMPONENT_SIZE}>
                    <Table dataSource={dataSource}
                           columns={columns}
                           pagination={{
                               total: dataSource !== undefined && dataSource !== null ? dataSource.length : 0,
                               showQuickJumper: true,
                               showSizeChanger: true,
                               pageSize: this.getFeaturePageSize(),
                               defaultPageSize: TABLE_ITEM_SIZE,
                               disabled: false,
                               hideOnSinglePage: true,
                               onShowSizeChange: (current, pageSize) => {
                                   this.setFeaturePageSize(pageSize);
                               }
                           }}
                           size={COMPONENT_SIZE}/>
                </Card>
            </Col>
        </Row>
    }
}

export class CollinearityDetectionStep extends StepWithStdConfigAndStatusAndFeatureCard  {
    static getDisplayName() { return "Collinearity detection";}
    static getTypeName() { return  "MulticollinearityDetectStep"; }
    render() {
        const stepData = this.props.stepData;
        let dataSource;
        if (stepData.status === StepStatus.Finish) {
            dataSource = stepData.extension.unselected_features?.map((value, index, arr) => {
                return {
                    key: index,
                    removed: value.removed,
                    reserved: value.reserved,
                }
            });
        } else {
            dataSource = null;
        }

        const columns = [
            makeColumn('Removed', 'removed'),
            makeColumn('Reserved', 'reserved')
        ];

        return <Row gutter={[2, 2]}>
            {this.getBasicCards()}
            <Col span={11} offset={2}>
                <Card title="Removed features" bordered={false} style={{width: '100%'}} size={COMPONENT_SIZE}>
                    <Table dataSource={dataSource}
                           columns={columns}
                           size={COMPONENT_SIZE}
                           pagination={{pageSize: TABLE_ITEM_SIZE, hideOnSinglePage: true}}
                    />
                </Card>
            </Col>
        </Row>
    }
}

export class DriftDetectionStep extends StepWithStdConfigAndStatusAndFeatureCard {
    static getDisplayName() { return "Drift detection";}
    static getTypeName() { return  "DriftDetectStep"; }

    render() {
        const stepData = this.stepData;
        let driftFeatureAUCDataSource;
        if (stepData.status === StepStatus.Finish) {
            driftFeatureAUCDataSource = stepData.extension.drifted_features_auc?.map((value, index, arr) => {
                return {
                    key: index,
                    feature: value.feature,
                    score: formatFloat(value.score),
                }
            });
        } else {
            driftFeatureAUCDataSource = null;
        }

        const driftFeatureAUCColumns = [
            makeColumn('Feature', 'feature'),
            makeColumn('AUC', 'score')
        ];

        const removedFeaturesInEpochColumns = [
            makeColumn('Feature', 'feature'),
            makeColumn('Importance', 'importance')
        ];

        const removeByEvaluateResultTitle = <Tooltip
            title={"Labeling the training set and the test set with different labels respectively as traget, and use only one feature to train a model step by step, if the evaluation score of the model exceed the set threshold, indicate that the feature drifts"}>
            Remove by evaluation
        </Tooltip>;

        const removeByFeatureImportance = <Tooltip
            title={"Labeling the training set and the test set with different labels respectively as target column, and use all features to train a model,  if evaluation score of the model exceed the set threshold, indicate that the features drifts. According to the feature importance to remove some top features and use the remaining features to train again until the final model evaluation score no longer exceed the threshold or reach the minimum number of features."}>
            Remove by feature importance
        </Tooltip>;

        return <Row gutter={[2, 2]}>
            {this.getBasicCards()}
            <Col span={11} offset={2}>
                <Row gutter={[2, 2]}>
                    <Col span={24}>
                        <Card title={removeByEvaluateResultTitle} bordered={false} style={{width: '100%'}}
                              size={COMPONENT_SIZE}>
                            <Table dataSource={driftFeatureAUCDataSource}
                                   columns={driftFeatureAUCColumns}
                                   size={COMPONENT_SIZE}
                                   pagination={{pageSize: 7, hideOnSinglePage: true}}
                            />
                        </Card>
                    </Col>
                    <Col span={24}>
                        <Card title={removeByFeatureImportance} bordered={false} style={{width: '100%'}}
                              size={COMPONENT_SIZE}>
                            <Tabs defaultActiveKey="1" tabPosition={'top'} style={{height: '100%', width: '100%'}}>
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
                                                   pagination={{pageSize: TABLE_ITEM_SIZE, hideOnSinglePage: true}}
                                            />
                                        </TabPane>
                                    })
                                }
                            </Tabs>
                        </Card>
                    </Col>
                </Row>

            </Col>
        </Row>
    }
}

class BaseImportanceSelectionStep extends StepWithStdConfigAndStatusAndFeatureCard{

    constructor(props) {
        super(props);
        this.state = {
            featurePageSize: TABLE_ITEM_SIZE * 2
        }
    }

    getFeaturePageSize() {
        return this.state.featurePageSize
    };

    setFeaturePageSize(featurePageSize){
        this.setState({featurePageSize: featurePageSize})
    }

    render() {
        const stepData = this.stepData;
        let dataSource;
        if (stepData.status === StepStatus.Finish) {
            dataSource = stepData.extension.importances?.map((value, index, arr) => {
                return {
                    key: index,
                    featureName: value.name,
                    importance: value.importance,
                    dropped: value.dropped
                }
            });
        } else {
            dataSource = null;
        }

        const columns = [
            makeColumn('Feature', 'featureName'),
            {
                title: 'Importance',
                dataIndex: 'importance',
                key: 'importance',
                render: function (text, record, index) {
                    return formatFloat(record.importance)
                },
                defaultSortOrder: 'descend',
                sorter: (a, b) => a.importance - b.importance,

            }, {
                title: 'Status',
                dataIndex: 'dropped',
                key: 'dropped',
                render: function (text, record, index) {
                    if (record.dropped) {
                        return <Tag color="red">Removed</Tag>
                    } else {
                        return <Tag color="green">Reserved</Tag>
                    }
                }
            }
        ];

        return <Row gutter={[2, 2]}>
            {this.getBasicCards()}

            <Col span={11} offset={2}>
                <Card title="Importances" bordered={false} style={{width: '100%'}} size={COMPONENT_SIZE}>
                    <Table dataSource={dataSource}
                           columns={columns}
                           pagination={{
                               total: dataSource !== undefined && dataSource !== null ? dataSource.length : 0,
                               showQuickJumper: true,
                               showSizeChanger: true,
                               disabled: false,
                               pageSize: this.getFeaturePageSize(),
                               hideOnSinglePage: true,
                               onShowSizeChange: (current, pageSize) => {
                                   this.setFeaturePageSize(pageSize);
                               }
                           }}
                           size={COMPONENT_SIZE}/>
                </Card>
            </Col>
        </Row>
    }
}

class FeatureImportanceSelectionStep extends BaseImportanceSelectionStep{
    static getDisplayName() { return "Feature selection"; }
    static getTypeName() { return  "FeatureImportanceSelectionStep"; }

    static getConfigTip() {
        return {
            feature_reselection: "Whether to enable two stage feature selection with permutation importance.",
            estimator_size: "The number of estimator to evaluate feature importance. Only valid when *feature_reselection* is True.",
            threshold: "Confidence threshold of the mean permutation importance. Only valid when *feature_reselection_strategy* is 'threshold'.",
        }
    }
}

class PermutationImportanceSelectionStep extends BaseImportanceSelectionStep{
    static getDisplayName() { return 'Feature selection' + TWO_STAGE_SUFFIX }
    static getTypeName() { return  "PermutationImportanceSelectionStep"; }
    static getConfigTip() {
        return {
            estimator_size: "The number of estimator to evaluate feature importance. Only valid when *feature_reselection* is True.",
            strategy: "Strategy to reselect features(*threshold*, *number* or *quantile*).",
            threshold: "Confidence threshold of the mean permutation importance. Only valid when *feature_reselection_strategy* is 'threshold'.",
            quantile: "Confidence quantile of feature_importance. Only valid when *feature_reselection_strategy* is 'quantile'.",
            number: "Expected feature number to keep. Only valid when *feature_reselection_strategy* is 'number'."
        }
    }
    getConfigTip() { return  PermutationImportanceSelectionStep.getConfigTip(); }
}

export class PseudoLabelStep extends StepWithStdConfigAndStatusAndFeatureCard {

    static getDisplayName() { return 'Pseudo labeling' }
    static getTypeName() { return  "PseudoLabelStep"; }
    static getConfigTip() {
        return {
            proba_threshold: "Confidence threshold of pseudo-label samples. Only valid when *pseudo_labeling_strategy* is 'threshold'.",
            resplit: "Whether to re-split the training set and evaluation set after adding pseudo-labeled data. If False, the pseudo-labeled data is only appended to the training set. Only valid when *pseudo_labeling* is True.",
            strategy: "Strategy to sample pseudo labeling data(*threshold*, *number* or *quantile*)."
        }
    }
    getConfigTip() { return  PseudoLabelStep.getConfigTip(); }

    render() {
        const stepData = this.stepData;
        let selectedLabel;
        let samplesObj;
        let labels;
        if (stepData.status === StepStatus.Finish) {
            selectedLabel = stepData.extension.selectedLabel;
            samplesObj = stepData.extension.samples;
            labels = Object.keys(stepData.extension.samples);
        } else {
            selectedLabel = null;
            samplesObj = null;
            labels = [];
        }

        const getProbaDensityEchartOpts = (labelName) => {
            var X_data = [];
            var y_data = [];
            if (stepData.status === StepStatus.Finish) {
                const probabilityDensity = stepData.extension.probabilityDensity;
                if (!isEmpty(labelName)) {
                    const probabilityDensityLabelData = probabilityDensity[labelName];
                    const gaussianData = probabilityDensityLabelData['gaussian'];
                    X_data = gaussianData['X'];
                    y_data = gaussianData['probaDensity'];
                } else {
                    showNotification('labelName is null');
                }
            }

            return {
                xAxis: {
                    type: 'category',
                    boundaryGap: true,
                    data: X_data,
                    axisLabel: {
                        interval: X_data.length / 10,  // show only 10 labels
                        formatter: function (value, index) {
                            return (index / X_data.length).toFixed(2);  // scale x axis to [0, 1]
                        }
                    }
                },
                yAxis: {
                    type: 'value'
                },
                series: [{
                    data: y_data,
                    type: 'line',
                    areaStyle: {}
                }]
            };
        };


        const probaDensityChartOption = getProbaDensityEchartOpts(selectedLabel);

        const onLabelChanged = (value) => {
            this.props.dispatch(
                {
                    type: 'probaDensityLabelChange',
                    payload: {
                        stepIndex: stepData.index,
                        selectedLabel: value
                    }
                }
            )
        };

        const samplesDataSource = isEmpty(samplesObj) ? null : Object.keys(samplesObj).map((value, index, array) => {
            return {
                key: index,
                label: value,
                count: samplesObj[value]
            }
        });


        const samplesColumns = [
            makeColumn('Label', 'label'),
            makeColumn('Count', 'count')
        ];
        const {Option} = Select;
        return <Row gutter={[2, 2]}>
            {this.getBasicCards()}
            <Col span={10} offset={2}>
                <Row gutter={[2, 2]}>
                    <Col span={24} offset={0}>
                        <Card title="Density plot of probability" bordered={false} style={{width: '100%'}}
                              size={COMPONENT_SIZE}>
                                <span>
                                     <span style={{marginLeft: '10px', marginRight: '10px'}}>Select label:</span>
                                     <Select defaultValue={selectedLabel} value={selectedLabel} style={{width: '50%'}}
                                             onChange={onLabelChanged} disabled={isEmpty(selectedLabel)}>
                                        {
                                            isEmpty(labels) ? null : labels.map(v => {
                                                return <Option key={`opt_${v}`} value={v}>{v}</Option>
                                            })
                                        }
                                    </Select>
                                </span>
                            <EchartsCore option={probaDensityChartOption} prepare={echarts => {
                                echarts.use([LineChart, GridComponent, TooltipComponent]);
                            }}/>
                        </Card>
                    </Col>
                    <Col span={24} offset={0}>
                        <Card title="Number of pseudo labeled samples " bordered={false} style={{width: '100%'}}
                              size={COMPONENT_SIZE}>
                            <Table dataSource={samplesDataSource}
                                   columns={samplesColumns}
                                   pagination={false}
                                   size={COMPONENT_SIZE}/>
                        </Card>
                    </Col>
                </Row>
            </Col>
        </Row>
    }
}

export class DaskPseudoLabelStep extends PseudoLabelStep{
    static getDisplayName() { return 'Dask Pseudo labeling'; }
    static getTypeName() { return  "DaskPseudoLabelStep"; }
}

export class EnsembleStep extends StepWithStdConfigAndStatusAndFeatureCard {
    static getDisplayName() { return 'Ensemble'; }
    static getTypeName() { return  "EnsembleStep"; }

    static getConfigTip() {
        return {
            scorer: "Scorer to used for feature importance evaluation and ensemble."
        };
    }
    getConfigTip() { return  EnsembleStep.getConfigTip(); }

    render() {
        const stepData = this.stepData;
        const getLiftEchartOpts = () => {
            var scores = [];
            var yLabels = [];
            const marks = [{type: 'max', name: 'Maximum'}, {type: 'min', name: 'Minimum'}];
            // const marks = [];
            if (stepData.status === StepStatus.Finish) {
                const scores_ = stepData.extension.scores;
                if (scores_ !== undefined && scores_ !== null && scores_.length > 0) {
                    yLabels = Array.from({length: scores_.length}, (v, k) => k);
                    scores = [...scores_];
                    const _lastScoreIndex = scores.length - 1;
                    marks.push({
                        name: 'Final',
                        coord: [yLabels[yLabels.length - 1], scores[_lastScoreIndex]],
                        value: scores[_lastScoreIndex]
                    })
                }
            }
            return {
                tooltip: {
                    show: true
                },
                legend: {
                    data: []
                },
                grid: {
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    data: yLabels,
                    axisLabel: {
                        formatter: function (v) {
                            return `epoch-${v}`;
                        }
                    }
                },
                yAxis: {
                    type: 'value',
                },
                series: [{
                    name: 'Score',
                    data: scores,
                    type: 'line',
                    markPoint: {
                        data: marks,
                        label: {
                            formatter: function (v) {
                                return formatFloat(v.value, 2)
                            }
                        }
                    },
                    smooth: true
                }]
            };
        };

        const getWeightsEchartOpts = () => {
            var weights = [];
            var yLabels = [];
            if (stepData.status === StepStatus.Finish) {
                const weights_ = stepData.extension.weights;
                if (weights_ !== undefined && weights_ !== null && weights_.length > 0) {
                    yLabels = Array.from({length: weights_.length}, (v, k) => k);
                    weights = [...weights_]
                }
            }

            // const yLabels = weights !== null && weights !== undefined ?  Array.from({length:weights.length}, (v,k) => k) : [];
            return {
                tooltip: {
                    trigger: 'axis'
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
                    axisLabel: {
                        formatter: function (v) {
                            return `model-${v}`;
                        }
                    }
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

        if (stepData.status === StepStatus.Skip) {
            return <SkippedStepContent/>
        } else {
            return <>
                <Row gutter={[2, 2]}>
                     {this.getBasicCards( false)}
                    <Col span={11} offset={2}>
                        <Row gutter={[2, 2]}>
                            <Card title="Weight" bordered={false} style={{width: '100%'}} size={COMPONENT_SIZE}>
                                <EchartsCore option={getWeightsEchartOpts()} prepare={echarts => {
                                    echarts.use([BarChart, GridComponent, TooltipComponent, LineChart]);
                                }}/>
                            </Card>
                        </Row>
                        <Row gutter={[2, 2]}>
                            <Card title="Lifting" bordered={false} style={{width: '100%'}} size={COMPONENT_SIZE}>
                                <EchartsCore option={getLiftEchartOpts()} prepare={
                                    echarts => {
                                        echarts.use([BarChart, GridComponent, TooltipComponent, LineChart, MarkPointComponent]);
                                    }}/>
                            </Card>
                        </Row>
                    </Col>
                </Row>
            </>
        }
    }
}

export class DaskEnsembleStep extends EnsembleStep{
    static getDisplayName() { return 'Dask ensemble'; }
    static getTypeName() { return  "DaskEnsembleStep"; }
}

export class FinalTrainStep extends StepWithStdConfigAndStatusAndFeatureCard {
    static getDisplayName() { return 'Final train'; }
    static getTypeName() { return  "FinalTrainStep"; }
    render() {
        return <Row gutter={[2, 2]}>
                {this.getBasicCards()}
            </Row>
    }
}

class TrialChart extends React.Component {

    constructor(props) {
        super(props);
        this.echartsLib = echarts;
        this.echartsElement = null;
        // props.trials
        // props.onTrialClick = (trialNo) => {}
    }

    onChartClick(params){

        // 1. if (params.) componentType:series , seriesType: scatter should match
        const xAxisName = params.name ; // FIXME: this is a bug
        const trainNo = parseInt(xAxisName.substring(1, xAxisName.length));
        this.props.onTrialClick(trainNo, 0);

    }

    componentDidMount() {

        echarts.use([MarkPointComponent, LineChart, GridComponent, TooltipComponent, ToolboxComponent, LegendComponent, ScatterChart, BarChart, TitleComponent]);  // this should be above of init echarts

        const echartsObj = this.echartsLib.init(this.echartsElement, this.props.theme, this.props.opts);
        this.renderChart(echartsObj);
        echartsObj.on('click', this.onChartClick.bind(this));

        window.addEventListener('resize', () => {
            if (echartsObj) echartsObj.resize();
        });

        if (this.props.showLoading) {
            echartsObj.showLoading(this.props.loadingOption || null);
        } else {
            echartsObj.hideLoading();
        }
    }

    componentDidUpdate(prevProps) {  // 第二次更新时候执行了这个方法
        const echartsObj = this.echartsLib.getInstanceByDom(this.echartsElement);
        this.renderChart(echartsObj);
    }

    renderChart(echartsObj){  // 第二次更新时候执行了这个方法

        // 生成options
        // 生成x坐标轴数据
        const trials = this.props.trials;
        const xAxisData = trials.map(v => {
            return `#${v.trialNo}`
        });


        // 生成模型的 reward 数据
        var nModles = 1;
        if (this.props.stepConfig.cv === true) {
            nModles = this.props.stepConfig.num_folds;
        }

        const chartOptions = this.getChartOptions(xAxisData, trials, this.props.stepConfig.cv, this.props.stepConfig.num_folds);

        // fixme check echartsElement is not empty
        // const echartsObj = this.echartsLib.getInstanceByDom(this.echartsElement);
        console.info("Trails chartOptions");
        console.info(chartOptions);

        echartsObj.setOption(chartOptions, false, false);
    }
    componentWillUnmount(){
        this.dispose();
    }

    dispose = () => {
        if (this.echartsElement) {
            try {
                throttleClear(this.echartsElement);
            } catch (e) {
                console.warn(e);
            }
            this.echartsLib.dispose(this.echartsElement);
        }
    };

    render() {
        const { style, className } = this.props;
        const styleConfig = {
            height: 300,
            ...style,
        };
        return (
            <div
                ref={(e) => { this.echartsElement = e; }}
                style={styleConfig}
                className={className}
            />
        );
    }



    getChartOptions(xAxisData, trials, cv, num_folds){

        const maxElapsed = Math.max(...trials.map(v => v.elapsed));

        let timeMax ;
        let timeUnit;
        let elapsedSeriesData;
        if(maxElapsed < 60){
            timeMax = 60;
            timeUnit = 's';
            elapsedSeriesData = trials.map(v => v.elapsed);
        } else if(maxElapsed >= 60 && maxElapsed < 3600 ){
            timeMax = 60;
            timeUnit = 'min';
            elapsedSeriesData = trials.map(v => v.elapsed / 60);
        } else {
            timeUnit = 'hour';
            timeMax = Math.floor(maxElapsed / 3600) + 1;
            elapsedSeriesData = trials.map(v => v.elapsed / 3600 );
        }

        // [ [0.5,0.5,0.9], [0.5,0.5,0.9] ]
        const limitStrLen = (str, len=12, fromBegin=true) => {
            if(str.length > len){
                if(fromBegin){
                    return str.substring(0, len)
                }else{
                    return str.substring(str.length - len, str.length)
                }
            }else {
                return str;
            }
        };

        const getSelection = (name, paramsObj)=> {

            const rows = Object.keys(paramsObj).map(key => {
                return `<tr>
                    <td>${limitStrLen(key, 12, false)}: </td>
                    <td>${limitStrLen(paramsObj[key])}</td>
                </tr>`
            });

            const s = `<div><span style="font-weight:bold">${name}</span></div>
            <div>
            <table>
                ${rows.join('')}
            </table>
            </div>`;

            // console.log(s);
            return s;
        };

        const getTooltipBody = (trial)=> {

            const trialDetail = {
                "Reward": trial.reward,
                "Elapsed": formatHumanDate(trial.elapsed),
                "Trial no": trial.trialNo,
                // todo "Status": 'finish'
            };

            const trialSection = getSelection('Trial', trialDetail);
            const paramsSection = getSelection('Params', trial.hyperParams);

            return trialSection + `<br/>` + paramsSection;

        };

        const colors_bak = ['#EE6666', '#91CC75', '#5470C6'];
        const colors = ['#91CC75', '#5470C6'];

        return {
            color: colors,
            title: {
                // subtext: 'Trials'
            },
            tooltip: {
                trigger: 'item',
                axisPointer: {
                    type: 'cross'
                },
                padding: [
                    2,  // 上
                    2, // 右
                    2,  // 下
                    2, // 左
                ],
                position: function(point, params, dom, rect, size){
                    return {
                        left: point[0] + 5,
                        top: point[1] - 5
                    }
                },
                formatter(params){
                    if(params.componentType === 'markPoint'){
                        return `<span style="display:inline-block;margin-right:5px;border-radius:50%;width:10px;height:10px;left:5px;background-color: ${params.color}"></span> ${params.name}: ${params.value}`
                    }else {
                        const trialNo = parseInt(params.name.substring(1, params.name.length));
                        var body = '';
                        trials.forEach(trial => {
                            if(trial.trialNo === trialNo){
                                body = getTooltipBody(trial)
                            }
                        });
                        return body;
                    }
                }
            },
            grid: {
                right: '20%'
            },
            toolbox: {
            },
            legend: {
                // data: ['Reward', 'Elapsed']
                // data: []
            },
            xAxis: [
                {
                    type: 'category',
                    data: xAxisData
                }
            ],
            dataZoom: [
                {
                    type: 'inside',
                    start: 0,
                    end: 100
                },
                {
                    show: true,
                    type: 'slider',
                    top: '90%',
                    start: 50,
                    end: 100
                }
            ],
            yAxis: [
                {
                    type: 'value',
                    name: 'Reward',
                    min: 0,
                    max: 1,
                    position: 'left',
                    axisLine: {
                        show: true,
                        lineStyle: {
                            // color: colors[0],
                        }
                    },
                    axisLabel: {
                        formatter: '{value}'
                    }
                }, {
                    type: 'value',
                    name: 'Elapsed',
                    min: 0,
                    // max: timeMax,
                    position: 'right',
                    axisLine: {
                        show: true,
                        lineStyle: {
                            // color: colors[1],
                        }
                    },
                    axisLabel: {
                        formatter: `{value} ${timeUnit}`
                    }
                }
            ],
            series: [
                {
                    name: 'Reward',
                    type: 'line',
                    color: colors[0],
                    yAxisIndex: 1,
                    data: trials.map(t => parseFloat(t.reward)),
                    markPoint: {
                        symbol: 'pin',
                        data: [
                            {type: 'max', name: 'Maximum'},
                            {type: 'min', name: 'Minimum'},
                        ],
                        label:{
                            color: 'white',
                            formatter: function (v) {
                                return formatFloat(v.value, 2)
                            }
                        }
                    },
                    tooltip: {
                        trigger: 'item'
                    }
                },
                {
                    name: 'Elapsed',
                    type: 'bar',
                    color: colors[1],
                    data: elapsedSeriesData
                },
            ]
        };
    }

}

TrialChart.propTypes = {
    trials: PropTypes.array,
    stepConfig: PropTypes.object,
    showLoading: PropTypes.bool,
    loadingOption: PropTypes.object,
    theme: PropTypes.oneOfType([
        PropTypes.string,
        PropTypes.object,
    ]),

};

TrialChart.defaultProps = {
    trials: [],
    stepConfig: {},
    showLoading: false,
    loadingOption: null,
    theme: null,
};


/***
 *
 * @param importances  [[{'name': 'age', 'imp': 0.6}]]
 * @returns {*}
 * @constructor
 */
function ImportanceBarChart({importances}) {

    const features = [...importances[0].map(v => v.name)].reverse();
    const legends = [];
    const series = importances.map((value, index, array) => {
        let name;
        if(importances.length > 1){
            name = `fold-${index}`;
            legends.push(name);
        }else{
            name = 'Importances'
        }
        return {
            name: name,
            type: 'bar',
            data: [...value.map(t => t.imp)].reverse()
        }
    });


    const featureImportanceChartOption = {
        title:{
            // text: 'feature importance',
            // todo subtext: 'Importance',
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        legend: {
            data: legends
        },
        grid: {
            // left: '3%',
            // right: '4%',
            // bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'value',
            boundaryGap: [0, 0.01]
        },
        yAxis: {
            type: 'category',
            data: features,
            axisLabel: {
                rotate:45,
                formatter: function (value, index) {
                    const label = value.toString();
                    const maxlength = 8;
                    if (label.length>maxlength) {
                        return label.substring(0, maxlength-3)+'...';
                    } else {
                        return label;
                    }
                }
            }
        },
        series: series
    };
    return <EchartsCore option={featureImportanceChartOption}/>

}

function CircleProgress({title, style, strokeColor, data}) {
    const {percent, value, tip} = data;
    return <Tooltip title={tip}>
        <Progress type="circle"
                  width={70}
                  strokeColor={strokeColor}
                  percent={percent}
                  style={{...style}}
                  format={percent => {
                      return <><div style={{fontSize: 10}}>
                          {title}
                      </div >
                          <div style={{fontSize: 13}} >{value}</div>
                      </>
                  }} />
    </Tooltip>
}

export class PipelineOptimizationStep extends BaseStepComponent {

    static getDisplayName() { return 'Pipeline optimization'; }
    static getTypeName() { return  "SpaceSearchStep"; }
    static getConfigTip(){
        return {
            cv: "If True, use cross-validation instead of evaluation set reward to guide the search process",
            num_folds: "Number of cross-validated folds, only valid when cv is true"
        }
    }
    getConfigTip(){
        return PipelineOptimizationStep.getConfigTip();
    }
    getDisplayConfigData() {
        const outputs = {...this.stepData.configuration,
            "earlyStoppingEnable": this.stepData.configuration.earlyStopping.enable};
        delete outputs['earlyStopping'];  // do not display es
        return getConfigData(outputs, this.getConfigTip());
    }

    render() {
        const stepData = this.stepData;
        // const [trialsProcessData, setTrialsProcessData] = useState({percent: 0, value: '-', tip: 'Waiting'});
        // const [earlyStoppingRewardData, setEarlyStoppingRewardData] = useState({percent: 0, value: '-', tip: 'Waiting'});
        // const [earlyStoppingTrialsData, setEarlyStoppingTrialsData] = useState({percent: 0, value: '-', tip: 'Waiting'});
        // const [earlyStoppingElapsedTimeData, setEarlyStoppingElapsedTimeData] = useState({percent: 0, value: '-', tip: 'Waiting'});

        const ES_EMPTY = {percent: 0, value: '-', tip: 'Empty data'};
        const ES_DISABLED = {percent: 0, value: '-', tip: 'EarlyStopping is disabled '};

        const earlyStoppingConfig = stepData.configuration.earlyStopping;
        const earlyStoppingStatus = stepData.extension.earlyStopping;

        const trailsData = stepData.extension.trials;
        const selectedTrialNo = getOrDefault(stepData.extension.selectedTrialNo, -1);
        let importanceData;
        let lastTrial;
        if(trailsData !== undefined && trailsData !== null && trailsData.length > 0){
            lastTrial =  trailsData[trailsData.length-1];
            var selectedTrailData = null ;
            if(selectedTrialNo !== -1){
                trailsData.forEach(trial => {
                    if(trial.trialNo === selectedTrialNo){
                        selectedTrailData = trial;
                    }
                });
            }else{
                selectedTrailData = trailsData[trailsData.length-1];
            }
            if(notEmpty(selectedTrailData)){
                importanceData = selectedTrailData.models.map(m => m.importances)
            }else {
                importanceData = [[]]
            }
        } else {
            lastTrial = null;
            importanceData = [[]]
        }
        //

        const configurationForPanel =  this.getDisplayConfigData();

        const getESData = (func, earlyStoppingConfig, earlyStoppingStatus) => {
            // check enabled es
            const {enable} = earlyStoppingConfig;
            if(enable !== true){
                return ES_DISABLED;
            }
            if(earlyStoppingStatus !== undefined && earlyStoppingStatus !== null){
                return func(earlyStoppingConfig, earlyStoppingStatus)
            }else{
                // es data may be null
                return ES_EMPTY
            }
        };

        const getEarlyStoppingTrialsData = (earlyStoppingConfig, earlyStoppingStatus) => {
            const { maxNoImprovedTrials } = earlyStoppingConfig;
            const { counterNoImprovementTrials } = earlyStoppingStatus;
            let percent;
            let tip;
            let value;
            if(maxNoImprovedTrials !== undefined && maxNoImprovedTrials !== null && maxNoImprovedTrials > 0){
                // early stopping by trials num  is opening
                percent = (counterNoImprovementTrials / maxNoImprovedTrials) * 100;
                value = counterNoImprovementTrials;
                tip =  `Max no improved trials is ${maxNoImprovedTrials}, now is ${value}`;
            }else{
                percent = 0;
                value = '-';
                tip = `This strategy is off`;
            }
            return {percent, value, tip}
        };
        const getEarlyStoppingRewardData = (earlyStoppingConfig, earlyStoppingStatus) => {

            // check enabled es
            const {enable} = earlyStoppingConfig;
            if(enable !== true){
                let maxReward;
                // use trials reward
                if(trailsData !== undefined && trailsData !== null && trailsData.length > 0){
                    const reward = Math.max(...trailsData.map( v => v.reward));
                    if(isNaN(reward)){
                        maxReward = 'NaN'
                    }else{
                        maxReward = formatFloat(reward)
                    }
                }else{
                    maxReward = 'null'
                }
                return {percent: 0, value: maxReward, tip: 'Max reward'}
            }
            if(earlyStoppingStatus !== undefined && earlyStoppingStatus !== null){
                const {exceptedReward, mode} = earlyStoppingConfig;
                const {bestReward } = earlyStoppingStatus;
                let percent;
                let tip;
                let value;
                if(exceptedReward !== undefined && exceptedReward !== null){
                    // early stopping is opening
                    if(mode === 'max'){
                        if(bestReward !== undefined && bestReward !== null){
                            percent = (bestReward / exceptedReward) * 100;
                            value = formatFloat(bestReward, 4)
                        }
                    }else{
                        percent = 0;
                        if(bestReward !== undefined && bestReward !== null){
                            value = bestReward;
                        }else{
                            value = '-';
                        }
                    }
                    tip = `Excepted reward is ${exceptedReward}, now best reward is ${value}`;
                }else{
                    percent = 0;
                    value = '-';
                    tip = `This strategy is off`;
                }
                return {percent, value, tip}
            }else{
                // es data may be null
                return ES_EMPTY
            }
        };
        const getEarlyStoppingElapsedTimeData = (earlyStoppingConfig, earlyStoppingStatus) => {
            const { timeLimit } = earlyStoppingConfig;
            const { elapsedTime } = earlyStoppingStatus;
            let percent;
            let tip;
            let value;
            if(timeLimit !== undefined && timeLimit !== null && timeLimit > 0){
                // early stopping by trials num  is opening
                percent = (elapsedTime / timeLimit) * 100;
                value = formatHumanDate(elapsedTime);
                tip =  `Limited time is ${formatHumanDate(timeLimit)}, now running for ${value}`;
            } else {
                percent = 0;
                value = '-';
                tip = `This strategy is off`;
            }
            return {percent, value, tip}
        };

        const getTrialsProcessData = (maxTrials, lastTrial) => {
            // const finishedTrials = lastTrial.trialNo ;
            // const maxTrials = stepData.extension.maxTrials

            if(maxTrials !== undefined && maxTrials !== null ){
                if(lastTrial !== undefined && lastTrial !== null){
                    const finishedTrials = lastTrial.trialNo;
                    const tip = `Max trials is ${maxTrials}, we've searched ${finishedTrials} time(s)`;
                    return {percent: (finishedTrials / maxTrials)*100 , value: finishedTrials, tip: tip}
                }else{
                    return {percent: 0 , value: 0, tip: `Max trials is ${maxTrials}, but no finished trials yet`}
                }
            }else {
                return {percent: 0 , value: 0, tip: "maxTrials is null"}
            }
        };

        const earlyStoppingTrialsData = getESData(getEarlyStoppingTrialsData, earlyStoppingConfig, earlyStoppingStatus);
        const earlyStoppingRewardData = getEarlyStoppingRewardData(earlyStoppingConfig, earlyStoppingStatus);
        const earlyStoppingElapsedTimeData = getESData(getEarlyStoppingElapsedTimeData, earlyStoppingConfig, earlyStoppingStatus);
        const trialsProcessData = getTrialsProcessData(stepData.extension.maxTrials, lastTrial);

        const onTrialClick = (trialNo, modelIndex) => {
            this.props.dispatch({
                type: ActionType.FeatureImportanceChange,
                payload: {
                    stepIndex: stepData.index,
                    selectedTrialNo: trialNo
                }
            });
        };

        if(stepData.status === StepStatus.Skip) {
            return <SkippedStepContent />
        }else {
            return <>
                    <Row gutter={[2, 2]}>
                        <Col span={24}>
                            <Card title="Search trials" bordered={false} size={'small'}>
                                <Row>
                                    <Col span={12}>
                                        <TrialChart
                                            trials={trailsData}
                                            stepConfig={stepData.configuration}
                                            onTrialClick={onTrialClick}
                                        />
                                    </Col>
                                    <Col span={12} offset={0}>
                                        <ImportanceBarChart
                                            importances={importanceData}
                                        />
                                    </Col>
                                </Row>
                            </Card>
                        </Col>
                </Row>
                <Row gutter={[2, 2]}>
                    <Col span={11} offset={0}>
                        <ConfigurationCard configurationData={configurationForPanel}/>
                    </Col>
                    <Col span={11} offset={2}>
                        <Card title="Progress" bordered={false}
                              style={{width: '100%', marginRight: 0, paddingRight: 0}} size={'small'}>
                            <CircleProgress
                                title={'Trials'}
                                strokeColor='#2db7f5'
                                data={trialsProcessData}
                            />
                            <CircleProgress
                                title={'Reward'}
                                strokeColor='#6ca30f'
                                style={{marginLeft: '5%'}}
                                data={earlyStoppingRewardData}
                            />
                            <CircleProgress
                                title={'Not improve'}
                                strokeColor='#0e72cc'
                                style={{marginLeft: '5%'}}
                                data={earlyStoppingTrialsData}/>
                            <CircleProgress
                                title={'Time'}
                                strokeColor='#fa4343'
                                style={{marginLeft: '5%'}}
                                data={earlyStoppingElapsedTimeData}/>
                        </Card>
                    </Col>
                </Row>
                <Row gutter={[2, 2]}>
                    <Col span={11} offset={0}>
                        <StepStatusCard
                            stepData={stepData}
                        />
                    </Col>
                    <Col span={11} offset={2}>
                        <FeatureNumCard stepData={stepData} hasOutput={false}/>
                    </Col>
                </Row>
            </>
        }
    }
}

export function getStepComponent(stepType) {
    const stepClassList = [ DataCleaningStep,
                            FeatureGenerationStep,
                            CollinearityDetectionStep,
                            DriftDetectionStep,
                            PipelineOptimizationStep,
                            FeatureImportanceSelectionStep,
                            PermutationImportanceSelectionStep,
                            PseudoLabelStep,
                            DaskPseudoLabelStep,
                            FinalTrainStep,
                            EnsembleStep,
                            DaskEnsembleStep ];

    for (var cls of stepClassList){
        if (cls.getTypeName() === stepType){
            return cls;
        }
    }
    return  null;
}

