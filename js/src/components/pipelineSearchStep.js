import React, {useState} from "react";
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
    }

    componentDidMount() {

        echarts.use([LineChart, GridComponent, TooltipComponent, ToolboxComponent, LegendComponent, ScatterChart, BarChart]);  // this should be above of init echarts

        const echartsObj = this.echartsLib.init(this.echartsElement, this.props.theme, this.props.opts);

        const options = this.getChartOptions([], [], [], this.props.experimentConfig.cv, this.props.experimentConfig.nFolds);

        echartsObj.setOption(options, false, false);

        const onClickFunc = this.props.onClick;

        if(onClickFunc !== null && onClickFunc !== undefined){
            echartsObj.on('click', onClickFunc);
        };

        window.addEventListener('resize', () => {
            if (echartsObj) echartsObj.resize();
        });

        if (this.props.showLoading) {
            echartsObj.showLoading(this.props.loadingOption || null);
        } else {
            echartsObj.hideLoading();
        }
    }

    getChartOptions(xAxisData, elapsedSeriesData, modelsScore, cv, nFolds){
        // [ [0.5,0.5,0.9], [0.5,0.5,0.9] ]

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

    componentDidUpdate(prevProps) {  // 第二次更新时候执行了这个方法
        const prevNewTrialData = prevProps.newTrialData == null ? {}: prevProps.newTrialData;
        const thisNewTrialData = this.props.newTrialData;

        if(thisNewTrialData != null ) {
            if(prevNewTrialData.trialNo !== thisNewTrialData.trialNo){
                // try to update data
                this.props.trials.push(thisNewTrialData);
                // 生成options
                // 生成x坐标轴数据
                const trials = this.props.trials;
                const xAxisData = trials.map(v => {
                    return `#${v.trialNo}`
                });

                // 生成耗时的数据
                const elapseSeriesData = trials.map(value => {
                    return (value.elapsed / 60).toFixed(0)
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

                const chartOptions = this.getChartOptions(xAxisData, elapseSeriesData, rewardSeriesData, this.props.experimentConfig.cv, this.props.experimentConfig.nFolds);

                // fixme check echartsElement is not empty
                const echartsObj = this.echartsLib.getInstanceByDom(this.echartsElement);
                echartsObj.setOption(chartOptions, false, false);
            }
        }

    }

    componentWillUnmount() {
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
}

TrialChart.propTypes = {
    newTrialData: PropTypes.object,
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
    newTrialData: null,
    trials: [],
    experimentConfig: {},
    showLoading: false,
    loadingOption: null,
    theme: null,
};


function notEmpty(obj) {
    return obj !== undefined && obj !== null;
}

function ImportanceBarChart(props) {

    const {features, importances} = props;

    if(notEmpty(features)){
        if(notEmpty(importances)){
            if(features.length !== importances.length){
                showNotification('Draw feature importance error: features.length !== importances.length');
            } else {
                const featureImportanceChartOption = {
                    title:{
                        // text: 'feature importance',
                        // subtext: 'feature importance',
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
                            data: importances
                        }
                    ]
                };
                return <EchartsCore option={featureImportanceChartOption}/>
            }
        } else {
            showNotification('Draw feature importance error: importances is empty.');
        }
    } else {
        showNotification('Draw feature importance error: features is empty.');
    }
}


export function PipelineOptimizationStep({stepData, newTrialData = {}}){

    const [importanceData, setImportanceData] = useState();

    const trailsData = stepData.extension?.trials;

    console.info("newTrialData in PipelineOptimizationStep");
    console.info(newTrialData);

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


    const onTrialPointClick = (params) => {
        console.info(params);
        const SERIES_PREFIX = 'fold-';
        // update importance charts
        // stepData.extension.trials
        // const currentTrial = stepData.extension.trials[params.dataIndex]
        // 1. if (params.) componentType:series , seriesType: scatter should match
        const seriesName = params.name ; // FIXME: this is a bug
        if(seriesName.startsWith(SERIES_PREFIX)
            && params.componentType === 'series'
            && params.seriesType === 'scatter'){
            // 2. get dataIndex to retrieve trail detail
            const trialNo = params.dataIndex;
            const trialData = trailsData[trialNo-1];
            // 3. get model index from series
            const seriesName = params.seriesName;
            if(seriesName.startsWith(SERIES_PREFIX)){
                const foldNo = parseInt(seriesName.substring(SERIES_PREFIX.length, seriesName.length));
                const importanceData = trialData.models[foldNo - 1].feature_importance;
                // 7. refresh importance charts
                setImportanceData(importanceData);
            }
        }else{
            console.info("Only model point take effect. ");
        }
    };

    return <><Row gutter={[2, 2]}>
        <Col span={24} >
            <Card title="Pipeline optimization" bordered={ false } style={{ width: '100%' }}>
                <Row>
                    <Col span={10} >
                        <TrialChart
                            newTrialData={newTrialData}
                            experimentConfig={{cv: false ,nFolds: 0}}  />
                    </Col>
                    <Col span={10} offset={0}>
                        {/*<EchartsCore*/}
                        {/*    // loadingOption={{ color: '#1976d2' }}*/}
                        {/*    option={featureImportanceChartOption}*/}

                        {/*    // showLoading={loading}*/}
                        {/*    // style={style}*/}
                        {/*    // className={className}*/}
                        {/*/>*/}
                        <ImportanceBarChart
                            features={['a', 'b']}
                            importances={[0.1, 0.2]}
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

