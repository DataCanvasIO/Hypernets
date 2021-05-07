import  { useState } from  "react";
import {Card, Col, Form, Row, Switch, Table, Tabs, Radio} from "antd";
import * as echarts from 'echarts';

const { TabPane } = Tabs;


export function TrialChart() {

    const colors = ['#5470C6', '#91CC75', '#EE6666'];

    const option = {
        color: colors,
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross'
            }
        },
        grid: {
            right: '20%'
        },
        toolbox: {
            feature: {
                dataView: {show: true, readOnly: false},
                restore: {show: true},
                saveAsImage: {show: true}
            }
        },
        legend: {
            data: ['耗时', '指标']
        },
        xAxis: [
            {
                type: 'category',
                axisTick: {
                    alignWithLabel: true
                },
                data: ['#1', '#2', '#3', '#4', '#5', '#6', '#7', '#8', '#9', '#10', '#11', '#12']
            }
        ],
        yAxis: [
            {
                type: 'value',
                name: '耗时',
                min: 0,
                max: 30,
                position: 'right',
                axisLine: {
                    show: true,
                    lineStyle: {
                        color: colors[2]
                    }
                },
                axisLabel: {
                    formatter: '{value} min'
                }
            },
            {
                type: 'value',
                name: '指标',
                min: 0,
                max: 1,
                position: 'left',
                axisLine: {
                    show: true,
                    lineStyle: {
                        color: colors[0]
                    }
                },
                axisLabel: {
                    formatter: '{value}'
                }
            }
        ],
        series: [
            {
                name: '耗时1',
                type: 'bar',
                color: '4F69BB',
                data: [20, 25, 26, 23, 21, 18, 25, 15, 17, 20, 19, 20]
            },
            {
                name: '耗时2',
                type: 'bar',
                color: '4F69BB',
                data: [20, 25, 26, 23, 21, 18, 25, 15, 17, 20, 19, 20]
            },
            {
                name: '耗时3',
                type: 'bar',
                color: '4F69BB',
                data: [20, 25, 26, 23, 21, 18, 25, 15, 17, 20, 19, 20]
            },
            {
                name: 'fold1',
                type: 'scatter',
                color: '9EDF81',
                yAxisIndex: 1,
                data: [0.01, 0.02, 0.2, 0.3, 0.6, 0.6, 0.6, 0.63, 0.62, 0.63, 0.65, 0.69]
            },
            {
                name: 'fold2',
                type: 'scatter',
                color: '9EDF81',
                yAxisIndex: 1,
                data: [0.1, 0.2, 0.3, 0.4, 0.8, 0.8, 0.8, 0.83, 0.82, 0.83, 0.85, 0.89]
            },
            {
                name: 'fold3',
                type: 'scatter',
                color: '9EDF81',
                yAxisIndex: 1,
                data: [0.2, 0.3, 0.4, 0.5, 0.9, 0.9, 0.9, 0.87, 0.87, 0.87, 0.86, 0.91]
            },
            {
                name: 'avg',
                type: 'line',
                color: 'red',
                yAxisIndex: 1,
                data: [0.15, 0.23, 0.5, 0.5, 0.9, 0.9, 0.9, 0.87, 0.87, 0.87, 0.86, 0.91]
            },
        ]
    };
    var chartDom = document.getElementById('main');
    var myChart = echarts.init(chartDom);

}
export function ConfigurationCard({configurationData}) {
    return <Form
        labelCol={{ span: 10 }}
        wrapperCol={{ span: 3 }}
        layout="horizontal">
        {
            Object.keys(configurationData).map(key => {
                const v = configurationData[key];
                if (v === undefined || v === null){
                    return <Form.Item label={key}>
                        <span>None</span>
                    </Form.Item>
                }
                if((typeof  v)  === "boolean"){
                    return <Form.Item label={key}>
                        <Switch disabled checked/>
                    </Form.Item>
                }else{
                    return <Form.Item label={key}>
                        <span>{v}</span>
                    </Form.Item>
                }
            })
        }

    </Form>


}

export function DataCleaningStep({stepData}) {

    const dataSource = stepData.extension.dropped_columns?.map((value, index, arr) => {
        return {
            key: index,
            feature_name: value.name,
            reason: value.reason,
        }
    });

    const columns = [
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

    return <Row gutter={[4, 4]}>
        <Col span={10} >
            <Card title="Data cleaning configuration" bordered={false} style={{ width: '100%' }}>
                {
                    <ConfigurationCard configurationData={stepData.configuration}/>
                }
            </Card>
        </Col>

        <Col span={10} offset={2} >
            <Card title="Removed features" bordered={false} style={{ width: '100%' }}>
                <Table dataSource={dataSource} columns={columns} />
            </Card>
        </Col>

    </Row>

}


export function CollinearityDetectionStep({stepData}){
    const dataSource = stepData.extension.dropped_columns?.map((value, index, arr) => {
        return {
            key: index,
            removed: value.removed,
            reserved: value.reserved,
        }
    });
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

    return <Row gutter={[4, 4]}>
        <Col span={10} >
            <Card title="Collinearity detection configuration" bordered={false} style={{ width: '100%' }}>
                {
                    <ConfigurationCard configurationData={stepData.configuration}/>
                }
            </Card>
        </Col>

        <Col span={10} offset={2} >
            <Card title="Removed features" bordered={false} style={{ width: '100%' }}>
                <Table dataSource={dataSource} columns={columns} />
            </Card>
        </Col>

    </Row>
}

export function DriftDetectionStep({stepData}){


    const driftFeatureAUCColumns = [
        {
            title: 'Feature',
            dataIndex: 'feature',
            key: 'feature',
        },
        {
            title: 'AUC',
            dataIndex: 'score',
            key: 'score',
        }
    ];

    const driftFeatureAUCDataSource = stepData.extension.drifted_features_auc?.map((value, index, arr) => {
        return {
            key: index,
            feature: value.feature,
            score: value.score,
        }
    });

    const removedFeaturesInEpochColumns = [
        {
            title: 'Feature',
            dataIndex: 'feature',
            key: 'feature',
        },
        {
            title: 'Importance',
            dataIndex: 'importance',
            key: 'importance',
        }
    ];

    return <><Row gutter={[4, 4]}>
        <Col span={10} >
            <Card title="Collinearity detection configuration" bordered={false} style={{ width: '100%' }}>
                {
                    <ConfigurationCard configurationData={stepData.configuration}/>
                }
            </Card>
        </Col>

        <Col span={10} offset={2} >
            <Card title="Removed features in epochs" bordered={false} style={{ width: '100%' }}>
                <Tabs defaultActiveKey="1" tabPosition={'top'} style={{ height: '100%', width: '100%'}}>
                    {
                        stepData.extension.removed_features_in_epochs?.map(epoch => {
                            return <TabPane tab={`Epoch ${epoch.epoch}`} key={epoch.epoch}>
                                <Table dataSource={epoch.removed_features?.map((value, index, arr) => {
                                    return {
                                        key: index,
                                        feature: value.feature,
                                        importance: value.importance,
                                    }
                                })} columns={removedFeaturesInEpochColumns} />
                            </TabPane>
                        })
                    }
                </Tabs>
            </Card>
        </Col>

    </Row>
        <Row gutter={[4, 4]}>
            <Col span={10} offset={12} >
                <Card title="Drifted features AUC" bordered={false} style={{ width: '100%' }}>
                    <Table dataSource={driftFeatureAUCDataSource} columns={driftFeatureAUCColumns} />
                </Card>
            </Col>
        </Row>
        </>
}
