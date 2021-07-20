import React, {useEffect, useState} from "react";
import {Card, Col, Row, Table} from "antd";
import EchartsCore from "./echartsCore";
import {ConfigurationCard, getConfigData, SkippedStepContent, StepStatusCard} from "./steps";

import * as echarts from "echarts/lib/echarts";
import 'echarts/lib/chart/heatmap';
import PropTypes from 'prop-types';
import { clear } from 'echarts/lib/util/throttle';
import { LineChart } from 'echarts/charts';
import { GridComponent } from 'echarts/components';
import { MarkPointComponent, TooltipComponent, ToolboxComponent, LegendComponent, TitleComponent  } from 'echarts/components';
import { ScatterChart } from 'echarts/charts';
import { BarChart } from 'echarts/charts';
import 'echarts/lib/component/legend';
// 引入dataZoom
import 'echarts/lib/component/dataZoom';
import { Progress, Tooltip } from 'antd';
import {formatFloat, formatHumanDate} from "../util";
import {Steps, StepStatus} from "../constants";

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
                            {type: 'max', name: 'Max reward'},
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


function notEmpty(obj) {
    return obj !== undefined && obj !== null;
}

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
            data: features
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

export function PipelineOptimizationStep({stepData}){

    const [importanceData, setImportanceData] = useState([[]]);
    // const [trialsProcessData, setTrialsProcessData] = useState({percent: 0, value: '-', tip: 'Waiting'});
    // const [earlyStoppingRewardData, setEarlyStoppingRewardData] = useState({percent: 0, value: '-', tip: 'Waiting'});
    // const [earlyStoppingTrialsData, setEarlyStoppingTrialsData] = useState({percent: 0, value: '-', tip: 'Waiting'});
    // const [earlyStoppingElapsedTimeData, setEarlyStoppingElapsedTimeData] = useState({percent: 0, value: '-', tip: 'Waiting'});

    const ES_EMPTY = {percent: 0, value: '-', tip: 'Empty data'};
    const ES_DISABLED = {percent: 0, value: '-', tip: 'EarlyStopping is disabled '};

    const earlyStoppingConfig = stepData.configuration.earlyStopping;
    const earlyStoppingStatus = stepData.extension.earlyStopping;


    const trailsData = stepData.extension.trials;

    var lastTrial;
    if(trailsData !== undefined && trailsData !== null && trailsData.length > 0){
        lastTrial =  trailsData[trailsData.length-1]
    }else{
        lastTrial = null;
    }

    // SearchSpaceStep的配置数据要剔除EarlyStopping数据
    const configurationForPanelObj =  {...stepData.configuration,
        "earlyStoppingEnable": stepData.configuration.earlyStopping.enable};
    delete configurationForPanelObj['earlyStopping'];
    const configurationForPanel = getConfigData(configurationForPanelObj, Steps.SpaceSearch.configTip);

    useEffect(() => {
        if(lastTrial !== null){
            setImportanceData(lastTrial.models.map(m => m.importances));
        }
    }, [lastTrial]);

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
        }else{
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
    const earlyStoppingRewardData = getESData(getEarlyStoppingRewardData, earlyStoppingConfig, earlyStoppingStatus);
    const earlyStoppingElapsedTimeData = getESData(getEarlyStoppingElapsedTimeData, earlyStoppingConfig, earlyStoppingStatus);
    const trialsProcessData = getTrialsProcessData(stepData.extension.maxTrials, lastTrial);

    const onTrialClick = (trialNo, modelIndex) => {
        const trials = stepData.extension.trials;
        trials.forEach(trial => {
            if(trial.trialNo === trialNo){
                setImportanceData(trial.models.map(m => m.importances))
            }
        });
    };








    let featuresDataSource;
    if(stepData.status === StepStatus.Finish){
        featuresDataSource = stepData.extension.input_features.map((value, index, arr) => {
            return {
                key: index,
                name: value,
                index: index
            }
        });
    }else{
        featuresDataSource = null;
    }

    const featuresColumns = [
        {
            title: 'Name',
            dataIndex: 'name',
            key: 'name',
        }
    ];

    if(stepData.status === StepStatus.Skip) {
        return <SkippedStepContent />
    }else {
        return <><Row gutter={[2, 2]}>
            <Col span={22}>
                <Card title="Search trials" bordered={false} style={{width: '100%'}} size={'small'}>
                    <Row>
                        <Col span={10}>
                            <TrialChart
                                trials={trailsData}
                                stepConfig={stepData.configuration}
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
                <Col span={10} offset={0}>
                    <ConfigurationCard configurationData={configurationForPanel}
                                       configurationTip={Steps.SpaceSearch.configTip}/>
                </Col>
                <Col span={10} offset={0}>
                    <Card title="Search progress" bordered={false}
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
                <Col span={10} offset={0}>

                        {/*<Table dataSource={featuresDataSource}*/}
                        {/*       columns={featuresColumns}*/}
                        {/*       pagination={{defaultPageSize: 5, disabled: false, pageSize: 5}}*/}
                        {/*       showHeader={false}/>*/}
                        {/*<ConfigurationCard*/}
                        {/*    configurationData={{*/}
                        {/*    "Begin time": 10,*/}
                        {/*    "Finished time": 10,*/}
                        {/*    "Elapsed": 10,*/}
                        {/*    "Status": 10,*/}
                        {/*}} configurationTip={*/}
                        {/*    {*/}
                        {/*        "Begin time": 10,*/}
                        {/*        "Finished time": 10,*/}
                        {/*        "Elapsed": 10,*/}
                        {/*        "Status": 10,*/}
                        {/*    }*/}
                        {/*}/>*/}
                    <StepStatusCard
                        stepData={stepData}
                    />
                </Col>
            </Row>
        </>
    }
}

