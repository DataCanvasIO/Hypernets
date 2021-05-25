import React, {useEffect, useState} from "react";
import {Card, Col, Row} from "antd";
import EchartsCore from "./echartsCore";
import {ConfigurationCard} from "./steps";

import ReactDOM from 'react-dom';
import {createStore} from "redux";
import {connect, Provider} from "react-redux";

import * as echarts from "echarts/lib/echarts";
import 'echarts/lib/chart/heatmap';
import PropTypes from 'prop-types';
import { clear } from 'echarts/lib/util/throttle';
import { isEqual } from 'date-fns';
import { bind } from 'zrender/lib/core/util';
import { LineChart } from 'echarts/charts';
import { GridComponent } from 'echarts/components';
import { TooltipComponent, ToolboxComponent, LegendComponent } from 'echarts/components';
import { ScatterChart } from 'echarts/charts';
import { BarChart } from 'echarts/charts';
import 'echarts/lib/component/legend';
// 引入dataZoom
import 'echarts/lib/component/dataZoom';
import {showNotification} from "../pages/experiment";


const pick = (obj, keys) => {
    const t = {};
    keys.forEach(key => {
        t[key] = obj[key];
    });
    return t;
};

const experimentConfig4Step = {
    cv: true,
    nFolds: 3,
};

class TrialChart extends React.Component {

    constructor(props) {
        super(props);
        this.echartsLib = echarts;
        this.echartsElement = null;
        // props.trials
        // props.onTrialClick = (trialNo) => {}
    }

    onChartClick(params){

        const SERIES_PREFIX = 'fold-';

        // 1. if (params.) componentType:series , seriesType: scatter should match
        const seriesName = params.name ; // FIXME: this is a bug
        if(this.props.experimentConfig.cv === false){
            const trainNo = parseInt(seriesName.substring(1, seriesName.length));
            this.props.onTrialClick(trainNo, 0, this.props.experimentConfig.cv);
        }

        // if(seriesName.startsWith(SERIES_PREFIX) && params.componentType === 'series' && params.seriesType === 'scatter'){
        //     // 2. get dataIndex to retrieve trail detail
        //     const trialNo = params.dataIndex;
        //     const modelIndex = 0;
        //     // const trialData = trailsData[trialNo-1];
        //     // // 3. get model index from series
        //     // const seriesName = params.seriesName;
        //     // if(seriesName.startsWith(SERIES_PREFIX)){
        //     //     const foldNo = parseInt(seriesName.substring(SERIES_PREFIX.length, seriesName.length));
        //     //     const importanceData = trialData.models[foldNo - 1].feature_importance;
        //     //     // 7. refresh importance charts
        //     //     setImportanceData(importanceData);
        //     //     this.props.onTrialClick(trialNo, modelIndex);
        //     // }
        // }else{
        //     console.info("Only model point take effect. ");
        // }

    }

    componentDidMount() {

        echarts.use([LineChart, GridComponent, TooltipComponent, ToolboxComponent, LegendComponent, ScatterChart, BarChart]);  // this should be above of init echarts

        const echartsObj = this.echartsLib.init(this.echartsElement, this.props.theme, this.props.opts);
        this.renderChart(echartsObj);
        // const options = this.getChartOptions([], [], [], this.props.experimentConfig.cv, this.props.experimentConfig.nFolds);
        // echartsObj.setOption(options, false, false);
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

        // 生成耗时的数据
        const elapseSeriesData = trials.map(value => {
            return (value.elapsed / 60).toFixed(0);
        });

        // 生成模型的 reward 数据
        var nModles = 1;
        if (this.props.experimentConfig.cv === true) {
            nModles = this.props.experimentConfig.nFolds;
        }

        // 生成模型的分数数据
        const rewardSeriesData = Array.from({length: nModles}, (k, v) => v).map(i => {
            return trials.map(v => v.models[i].reward)
        });

        const chartOptions = this.getChartOptions(xAxisData, elapseSeriesData, rewardSeriesData, trials, this.props.experimentConfig.cv, this.props.experimentConfig.nFolds);

        // fixme check echartsElement is not empty
        // const echartsObj = this.echartsLib.getInstanceByDom(this.echartsElement);
        echartsObj.setOption(chartOptions, false, false);
    }
    componentWillUnmount(){
        this.dispose();
    }

    dispose = () => {
        if (this.echartsElement) {
            try {
                clear(this.echartsElement);
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
    getChartOptions(xAxisData, elapsedSeriesData, modelsScore, trials, cv, nFolds){
        // [ [0.5,0.5,0.9], [0.5,0.5,0.9] ]

        const getTooltipBody = (name, paramsObj)=> {

            const rows = Object.keys(paramsObj).map(key => {
                return `<tr>
                    <td>${key}: </td>
                    <td>${paramsObj[key]}</td>
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


        const scoreSeries = [];
        if(cv === false){
            scoreSeries.push( {
                name: 'Reward',
                type: 'line',
                color: '#9EDF81',
                yAxisIndex: 1,
                data: modelsScore[0]
            })
        }else{

            Array.from({length: nFolds}, (k, v)=>v).map(i=>{
                scoreSeries.push( {
                    name: `fold_${i}`,
                    type: 'scatter',
                    color: '#9EDF81',
                    yAxisIndex: 1,
                    data: scoreSeries.map( (scores) => scores[i])
                })
            });

            // calc avg
            scoreSeries.push({
                name: 'Average',
                type: 'line',
                color: 'red',
                yAxisIndex: 1,
                data: scoreSeries.map(v => v.avgReward)
            })}

        const colors = ['#5470C6', '#91CC75', '#EE6666'];

        return {
            color: colors,
            title: {subtext: 'Trials' },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross'
                },
                formatter(params){
                    const param = params[0];
                    const trialNo = parseInt(param.axisValue.substring(1, param.axisValue.length));
                    console.log('trialNo');
                    console.log(trialNo);
                    var body = '';
                    trials.forEach(trial => {
                        if(trial.trialNo === trialNo){
                            body = getTooltipBody('Params', trial.hyperParams)
                        }
                    });
                    return body;
                }
            },
            grid: {
                right: '20%'
            },
            toolbox: {

            },
            legend: {
                data: []
            },
            xAxis: [
                {
                    type: 'category',
                    // axisTick: {
                    //     alignWithLabel: true
                    // },
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
                    name: 'Elapsed',
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
                    name: 'Reward',
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
                ...scoreSeries,
                {
                    name: 'Elapsed',
                    type: 'bar',
                    color: '#4F69BB',
                    data: elapsedSeriesData
                },
            ]
        };
    }

}

TrialChart.propTypes = {
    trials: PropTypes.array,
    experimentConfig: PropTypes.object,
    showLoading: PropTypes.bool,
    loadingOption: PropTypes.object,
    theme: PropTypes.oneOfType([
        PropTypes.string,
        PropTypes.object,
    ]),

};

TrialChart.defaultProps = {
    trials: [],
    experimentConfig: {},
    showLoading: false,
    loadingOption: null,
    theme: null,
};


function notEmpty(obj) {
    return obj !== undefined && obj !== null;
}

function ImportanceBarChart({importances}) {

    const features = Object.keys(importances);

    const featureImportanceChartOption = {
        title:{
            // text: 'feature importance',
            subtext: 'Feature importance',
        },
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
            data: features
        },
        series: [
            {
                name: 'Importances',
                type: 'bar',
                data: Object.keys(importances).map(v => importances[v])
            }
        ]
    };
    return <EchartsCore option={featureImportanceChartOption}/>

}


export function PipelineOptimizationStep({stepData}){

    const [importanceData, setImportanceData] = useState({});

    const trailsData = stepData.extension?.trials;

    useEffect(() => {
        if(trailsData !== undefined && trailsData !== null && trailsData.length > 0){
            setImportanceData(trailsData[trailsData.length-1].models[0].importances);
        }
    }, [trailsData]);


    // elapsedTime
    // trailsData
    // 1. generate cv series
    if (stepData.configuration.cv === true){
        // 1.1. cv models
        // const cvModelSeries  = Array.from({length: stepData.configuration.num_folds}).map(i => {
        //     return {
        //         name: `fold-${i}`,
        //         type: 'scatter',
        //         color: '#9EDF81',
        //         yAxisIndex: 1,
        //         data: trailsData.map(v => {v.models[i].reward_score})
        //     }
        // });

        // 1.2. average
        // trailsData.map( v => {
        //      return {
        //          name: 'avg',
        //          type: 'line',
        //          color: 'red',
        //          yAxisIndex: 1,
        //          data: v.models.map( m => m.reward_score).reduce((a, b) => a+b) / stepData.configuration.num_folds
        //      }
        // })

    }
    // 2.

    const  b = {'a': 0.1, 'b': 0.2 };

    const onTrialClick = (trialNo, modelIndex, cv) => {
        const trials = stepData.extension.trials;
        if(cv === false){
            trials.forEach(trial => {
                if(trial.trialNo === trialNo){
                    setImportanceData(trial.models[0].importances)
                }
            });
        }
        // const SERIES_PREFIX = 'fold-';
        // update importance charts
        // stepData.extension.trials
        // const currentTrial = stepData.extension.trials[params.dataIndex]
        // 1. if (params.) componentType:series , seriesType: scatter should match
        // 2. get dataIndex to retrieve trail detail
        // const trialNo = params.dataIndex;
        // const trialData = trailsData[trialNo-1];
        // // 3. get model index from series
        // const seriesName = params.seriesName;
        // if(seriesName.startsWith(SERIES_PREFIX)){
        //     const foldNo = parseInt(seriesName.substring(SERIES_PREFIX.length, seriesName.length));
        //     const importanceData = trialData.models[foldNo - 1].feature_importance;
        //     // 7. refresh importance charts
        //     setImportanceData(importanceData);
        // }

    };

    return <><Row gutter={[2, 2]}>
        <Col span={24} >
            <Card title="Pipeline optimization" bordered={ false } style={{ width: '100%' }}>
                <Row>
                    <Col span={10} >
                        <TrialChart
                            trials={trailsData}
                            experimentConfig={{cv: false ,nFolds: 0}}
                            onTrialClick={onTrialClick}
                        />
                    </Col>
                    <Col span={10} offset={0}>
                        <ImportanceBarChart
                            importances={importanceData}
                        />
                    </Col>
                </Row>
            </Card>
        </Col>

    </Row>
        <Row gutter={[2, 2]}>
            <Col span={10} offset={0} >
                <Card title="Pipeline optimization configuration" bordered={false} style={{ width: '100%' }}>
                    {
                        <ConfigurationCard configurationData={stepData.configuration}/>
                    }
                </Card>
            </Col>
        </Row>
    </>
}

