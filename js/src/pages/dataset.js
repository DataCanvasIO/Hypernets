import * as React from "react";
import './exploreDataset.css';

import {Card, Col,  Row, Table, Tooltip} from "antd";

import EchartsCore from "../components/echartsCore";
import {isEmpty} from "../util";
import { TooltipComponent } from 'echarts/components';
import { PieChart } from 'echarts/charts';
import { GridComponent } from 'echarts/components';
import { BarChart } from 'echarts/charts';



function FeatureBar({first, latest, color, typeName, nFeatures, percent}) {

    let borderRadius ;
    if(first){
        if(latest){
            borderRadius = {borderRadius: '20px 20px 20px 20px'};
        }else{
            borderRadius = {borderRadius: '20px 0 0 20px'};
        }
    }else{
        if(latest){
            borderRadius = {borderRadius: '0px 20px 20px 0'};
        }else{
            borderRadius = {}
        }
    }

    return <div  style={{ width: `${percent}%`, backgroundColor: color, ...borderRadius}}>
        {nFeatures} columns({percent}%)
    </div>
}

function FeatureLegend({typeName, nFeatures, percent, color}) {
    const tipContent = `${nFeatures} columns，${percent}%`;
    return <div className={'legendItem'}>
        <span className={'legendPoint'} style={{backgroundColor: color}}/>
        <Tooltip title={tipContent}>
            <span className={'symbol'}> {typeName} </span>
        </Tooltip>
        {/*{<span className={'comment'}> {nFeatures} columns，{percent}% </span>}*/}
    </div>
}

export function FeatureDistributionBar({data}) {

    const { nContinuous, nCategorical, nDatetime, nText, nLocation, nOthers } = data;
    const total = nContinuous + nCategorical + nDatetime + nText + nLocation + nOthers;

    const barList = [
        {
            typeName: "Continuous",
            color: "rgb(0, 183, 182)",
            nFeatures: nContinuous
        },
        {
            typeName: "Categorical",
            color: "rgb(0, 156, 234)",
            nFeatures: nCategorical
        },
        {
            typeName: "Datetime",
            color: "rgb(244, 148, 49)",
            nFeatures: nDatetime
        },
        {
            typeName: "Location",
            color: "rgb(88,125, 49)",
            nFeatures: nLocation
        },
        {
            typeName: "Text",
            color: "rgb(125, 0, 249)",
            nFeatures: nText
        },
        {
            typeName: "Other",
            color: "rgb(105, 125, 149)",
            nFeatures: nOthers
        }
    ];
    // 数据不能为空，
    // find first not empty
    var firstNonEmpty  = null ;
    var latestNonEmpty  = null ;
    for (var i = 0 ; i < barList.length ; i++){
        const item = barList[i];
        if(firstNonEmpty === null){
            if(item.nFeatures > 0){
                firstNonEmpty = i;
            }
        }
        const backIndex = barList.length - i - 1;
        const backItem = barList[backIndex];
        if(latestNonEmpty === null){
            if(backItem.nFeatures > 0){
                latestNonEmpty = backIndex;
            }
        }
    }

    if(firstNonEmpty !== null && latestNonEmpty !== null){
        const bars = [];
        const legends = [];

        barList.map((value, index , array) => {
            const percent =  ((value.nFeatures / total) * 100).toFixed(0);
            bars.push(
                <FeatureBar
                    key={value.typeName}
                    typeName={value.typeName}
                    first={index === firstNonEmpty}
                    latest={index === latestNonEmpty}
                    nFeatures={value.nFeatures}
                    color={value.color}
                    percent={percent}
            />);
            legends.push(
                <FeatureLegend
                    key={value.typeName}
                    typeName={value.typeName}
                    nFeatures={value.nFeatures}
                    percent={percent}
                    color={value.color}
                />
            )
        });

        return <>
            <div className={'bar'}>
                {bars}
            </div>
            <div className={'legend'}>
                {legends}
            </div>
            </>
    }else{
        return <span>
            Error, features is empty.
        </span>
    }
}


export function Dataset({data}){

    const targetData = data.target;
    const taskType = data.target.taskType;
    const displayKeyMapping = {
        name: 'Name',
        taskType: 'Task type',
        freq: "Freq",
        unique: "Unique",
        missing: "Missing",
        mean: "Mean",
        min: "Min",
        max: "Max",
        stdev: "Stdev",
        dataType: 'Data type',
    };

    const dataSource = Object.keys(targetData).map(key=> {

        const v = targetData[key];
        let displayValue ;
        if(isEmpty(v)){
            displayValue = '-';
        }else{
            if(!isNaN(v)){
                if(Number.isInteger(v)){
                    displayValue = v;
                }else{
                    displayValue = v.toFixed(4)
                }
            }else{
                displayValue = v;
            }
        }
        return {
            key: key,
            name: displayKeyMapping[key],
            value: displayValue
        }
    });
    const columns = [
        {
            title: 'Name',
            dataIndex: 'name',
            key: 'name',
        },
        {
            title: 'Value',
            dataIndex: 'value',
            key: 'value',
        }
    ];

    const dataShapeObj = data.datasetShape;

    const dataShapeDataSource = Object.keys(dataShapeObj).map(key=> {
        const v = dataShapeObj[key];
        let displayValue;
        if(isEmpty(v) || v.length < 1){
            displayValue = '-'
        }else{
            displayValue = `(${v.join(",")})`
        }
        return {
            key: key,
            name: key,
            value: displayValue
        }
    });
    /***
     *
     * @param targetDistribution
     *
     *  For continuous:
     *   {
     *       count: [1, 1, 1, 1, 1, 1, 1, 1, 19, 10],
     *       region: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5,6], [6, 7], [7, 8], [8, 9], [9, 10]]
     *   }
     *
     *   For categorical:
     *   {
     *       yes: 100,
     *       no: 20
     *   }
     * @param taskType
     * @returns {{yAxis: {type: string}, xAxis: {data: *, type: string}, series: [{data: *, type: string}]}|{legend: {orient: string, top: number, data: ([string, string]|string[]), bottom: number, right: number, type: string, selected: *}, series: [{data: [{name: string, value: number}, {name: string, value: number}]|({name: string, value: number}|{name: string, value: number})[], center: [string, string], name: string, emphasis: {itemStyle: {shadowOffsetX: number, shadowBlur: number, shadowColor: string}}, type: string, radius: string}], tooltip: {formatter: string, trigger: string}}}
     */
    const getTargetDistributionOption = (targetDistribution, taskType) => {
        let option;
        if(taskType !== 'regression'){
            const legendData = Object.keys(targetDistribution);
            const seriesData = Object.keys(targetDistribution).map(v => {
                return {
                    "name": v,
                    "value": targetDistribution[v]
                }
            });
            option = {
                tooltip: {
                    trigger: 'item',
                    formatter: '{a} <br/>{b} : {c} ({d}%)'
                },
                legend: {
                    type: 'scroll',
                    orient: 'vertical',
                    right: 50,
                    top: 100,
                    data: legendData
                },
                series: [
                    {
                        name: 'Label',
                        type: 'pie',
                        radius: '55%',
                        // center: ['40%', '50%'],
                        data: seriesData,
                        emphasis: {
                            itemStyle: {
                                shadowBlur: 10,
                                shadowOffsetX: 0,
                                shadowColor: 'rgba(0, 0, 0, 0.5)'
                            }
                        }
                    }
                ]
            };
        }else{
            const xAxisData = targetDistribution.region.map(v => `[${v[0]}, ${v[1]})`);
            option = {
                tooltip: {
                    trigger: 'axis',
                    axisPointer: {
                        type: 'shadow'
                    }
                },
                xAxis: {
                    type: 'category',
                    data: xAxisData
                },
                yAxis: {
                    type: 'value'
                },
                series: [{
                    data: targetDistribution.count,
                    type: 'bar'
                }]
            };
        }
        return option;
    };

    const option = getTargetDistributionOption(data.targetDistribution, data.target.taskType);

    const getPrepareFunc = (taskType) => {
        let prepare;
        if (taskType === 'regression') {
            prepare = (echarts) => {
                echarts.use([BarChart, GridComponent, TooltipComponent]);  // this should be above of init echarts
            }
        }else{
            prepare = (echarts) => {
                echarts.use([PieChart, TooltipComponent]);
            }
        }
        return prepare;
    };




    return <>
       <Row gutter={[4, 4]} align={'bottom'}>
        <Col span={24} >
            <Card title="Feature types distribution" bordered={false} style={{ width: '100%' }}>
                <FeatureDistributionBar data={data.featureDistribution}/>
            </Card>
        </Col>
        </Row>

        <Row>
            <Col span={12} >
                <Card title="Target" bordered={false} style={{ width: '100%' }}>
                    <Table dataSource={dataSource} columns={columns} pagination={false} showHeader={false}/>
                </Card>
            </Col>
            <Col span={12} >
                <Row>
                    <Col span={24} >
                        <Card title="Distribution of y" bordered={false} style={{ alignContent: 'center',
                            width: '100%',
                            height: '100%'}}>
                            <EchartsCore option={option}
                                         prepare={getPrepareFunc(taskType)}
                                         style={ {width: 450, height: 300}  }/>
                        </Card>
                    </Col>
                    <Col span={24} offset={0} >
                        <Card title="Dataset shape" bordered={false} style={{ width: '100%' }}>
                            <Table dataSource={dataShapeDataSource} columns={columns} pagination={false} showHeader={false}/>
                        </Card>
                    </Col>
                </Row>
            </Col>
        </Row>
    </>
}