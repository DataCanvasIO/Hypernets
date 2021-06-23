import React, {useEffect, useState} from "react";
import {Card, Col, Row, Table} from "antd";
import EchartsCore from "./echartsCore";
import {ConfigurationCard} from "./steps";


import * as echarts from "echarts/lib/echarts";
import 'echarts/lib/chart/heatmap';
import PropTypes from 'prop-types';
import { clear } from 'echarts/lib/util/throttle';
import { LineChart } from 'echarts/charts';
import { GridComponent } from 'echarts/components';
import { TooltipComponent, ToolboxComponent, LegendComponent } from 'echarts/components';
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

        echarts.use([LineChart, GridComponent, TooltipComponent, ToolboxComponent, LegendComponent, ScatterChart, BarChart]);  // this should be above of init echarts

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

        // on resize
        // if (this.echartsElement) {
        //     bind(this.echartsElement, () => {
        //         try {
        //             echartsObj.resize()
        //         } catch (e) {
        //             console.warn(e)
        //         }
        //     })
        // }

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

        // 生成模型的分数数据
        const rewardSeriesData = Array.from({length: nModles}, (k, v) => v).map(i => {
            return trials.map(v => v.models[i].reward)
        });

        const chartOptions = this.getChartOptions(xAxisData, rewardSeriesData, trials, this.props.stepConfig.cv, this.props.stepConfig.num_folds);

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



    getChartOptions(xAxisData, modelsScore, trials, cv, num_folds){

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
                "Elapsed time": formatHumanDate(trial.elapsed),
                "Trial no": trial.trialNo,
                // todo "Status": 'finish'
            };

           const trialSection = getSelection('Trial', trialDetail);
           const paramsSection = getSelection('Params', trial.hyperParams);

           return trialSection + `<br/>` + paramsSection;

        };

        const scoreSeries = [];

        scoreSeries.push();

        const colors = ['#5470C6', '#91CC75', '#EE6666'];

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
                formatter(params){
                    const trialNo = parseInt(params.name.substring(1, params.name.length));
                    var body = '';
                    trials.forEach(trial => {
                        if(trial.trialNo === trialNo){
                            body = getTooltipBody(trial)
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
                data: ['Reward', 'Elapsed']
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
                    name: 'Elapsed',
                    min: 0,
                    max: timeMax,
                    position: 'right',
                    axisLine: {
                        show: true,
                        lineStyle: {}
                    },
                    axisLabel: {
                        formatter: `{value} ${timeUnit}`
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

                        }
                    },
                    axisLabel: {
                        formatter: '{value}'
                    }
                }
            ],
            series: [
                {
                    name: 'Reward',
                    type: 'line',
                    // color: '#f59311',
                    color: '#6ca30f',
                    yAxisIndex: 1,
                    data: trials.map(t => t.reward)
                },
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
    const [earlyStoppingRewardData, setEarlyStoppingRewardData] = useState({percent: 0, value: '-', tip: 'Waiting'});
    const [earlyStoppingTrialsData, setEarlyStoppingTrialsData] = useState({percent: 0, value: '-', tip: 'Waiting'});
    const [earlyStoppingElapsedTimeData, setEarlyStoppingElapsedTimeData] = useState({percent: 0, value: '-', tip: 'Waiting'});

    const PROCESS_EMPTY_DATA = {percent: 0, value: '-', tip: 'Empty data'};

    const trailsData = stepData.extension?.trials;

    var lastTrial;
    if(trailsData !== undefined && trailsData !== null && trailsData.length > 0){
        lastTrial =  trailsData[trailsData.length-1]
    }else{
        lastTrial = null;
    }

    // SearchSpaceStep的配置数据要剔除EarlyStopping数据
    const configurationForPanel =   {...stepData.configuration};
    delete configurationForPanel['earlyStopping'];


    useEffect(() => {
        if(lastTrial !== null){
            setImportanceData(lastTrial.models.map(m => m.importances));
            setEarlyStoppingRewardData(getEarlyStoppingRewardData(lastTrial));
            setEarlyStoppingTrialsData(getEarlyStoppingTrialsData(lastTrial));
            setEarlyStoppingElapsedTimeData(getEarlyStoppingElapsedTimeData(lastTrial));

        }
    }, [lastTrial]);

    const onTrialClick = (trialNo, modelIndex) => {
        const trials = stepData.extension.trials;
        trials.forEach(trial => {
            if(trial.trialNo === trialNo){
                setImportanceData(trial.models.map(m => m.importances))
            }
        });
    };

    const getEarlyStoppingRewardData = (lastTrial) => {

        const earlyStoppingConfig = lastTrial.earlyStopping.config;

        if(earlyStoppingConfig === null || earlyStoppingConfig === undefined){
            return PROCESS_EMPTY_DATA
        }

        const {exceptedReward, direction} = earlyStoppingConfig;

        const {reward } = lastTrial.earlyStopping.status;
        let percent;
        let tip;
        let value;
        if(exceptedReward !== undefined && exceptedReward !== null){
            // early stopping is opening
            if(direction === 'max'){
                if(reward !== undefined && reward !== null){
                    percent = (reward / exceptedReward) * 100;
                    value = formatFloat(reward, 4)
                }
            }else{
                percent = 0;
                if(reward !== undefined && reward !== null){
                    value = reward;
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

    const getEarlyStoppingTrialsData = (lastTrial) => {
        const earlyStoppingConfig = lastTrial.earlyStopping.config;
        if(earlyStoppingConfig === null || earlyStoppingConfig === undefined){
            return PROCESS_EMPTY_DATA
        }

        const { maxNoImprovedTrials } = earlyStoppingConfig;

        const { noImprovedTrials } = lastTrial.earlyStopping.status;
        let percent;
        let tip;
        let value;
        if(maxNoImprovedTrials !== undefined && maxNoImprovedTrials !== null && maxNoImprovedTrials > 0){
            // early stopping by trials num  is opening
            percent = (noImprovedTrials / maxNoImprovedTrials) * 100;
            value = noImprovedTrials;
            tip =  `Max no improved trials is ${maxNoImprovedTrials}, now is ${value}`;
        }else{
            percent = 0;
            value = '-';
            tip = `This strategy is off`;
        }
        return {percent, value, tip}
    };

    const getEarlyStoppingElapsedTimeData = (lastTrial) => {
        const earlyStoppingConfig = lastTrial.earlyStopping.config;

        if(earlyStoppingConfig === null || earlyStoppingConfig === undefined){
            return PROCESS_EMPTY_DATA
        }

        const { maxElapsedTime } = earlyStoppingConfig;

        const { elapsedTime } = lastTrial.earlyStopping.status;
        let percent;
        let tip;
        let value;
        if(maxElapsedTime !== undefined && maxElapsedTime !== null && maxElapsedTime > 0){
            // early stopping by trials num  is opening
            percent = (elapsedTime / maxElapsedTime) * 100;
            value = formatHumanDate(elapsedTime);
            tip =  `Limited time is ${formatHumanDate(maxElapsedTime)}, now running for ${value}`;
        }else{
            percent = 0;
            value = '-';
            tip = `This strategy is off`;
        }
        return {percent, value, tip}
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

    return <><Row gutter={[2, 2]}>
        <Col span={22} >
            <Card title="Pipeline optimization" bordered={ false } style={{ width: '100%' }} size={'small'}>
                <Row>
                    <Col span={10} >
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
        <Col span={10} offset={0} >
            <Card title={"Configuration"} bordered={false} style={{ width: '100%' }} size={'small'}>
                {
                    <ConfigurationCard configurationData={configurationForPanel} configurationTip={Steps.SpaceSearch.configTip}/>
                }
            </Card>
        </Col>
        <Col span={10} offset={0} >
            <Card title="Early stopping" bordered={false} style={{ width: '100%' , marginRight: 0, paddingRight: 0}} size={'small'}>
                    <CircleProgress
                        title={'Reward'}
                        strokeColor ='#6ca30f'
                        data={earlyStoppingRewardData}
                        />
                    <CircleProgress
                        title={'Trials'}
                        strokeColor ='#0e72cc'
                        style={{marginLeft: '5%'}}
                        data={earlyStoppingTrialsData}/>
                    <CircleProgress
                        title={'Time'}
                        strokeColor ='#fa4343'
                        style={{marginLeft: '5%'}}
                        data={earlyStoppingElapsedTimeData}/>
            </Card>
        </Col>
    </Row>
    <Row gutter={[2, 2]}>
        <Col span={10} offset={10} >
            <Card title="Input features" bordered={false} style={{ width: '100%' }} size={'small'} >
                <Table dataSource={featuresDataSource}
                       columns={featuresColumns}
                       pagination={ {defaultPageSize: 5, disabled: false, pageSize:  5}}
                       showHeader={false} />
            </Card>
        </Col>
    </Row>
    </>
}

