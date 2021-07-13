import React, { Component } from 'react';
import * as echarts from "echarts/lib/echarts";
import 'echarts/lib/chart/heatmap';
import PropTypes from 'prop-types';
import { clear } from 'echarts/lib/util/throttle';
import { isEqual } from 'date-fns';
import { bind } from 'zrender/lib/core/util';
import { LineChart } from 'echarts/charts';
import { GridComponent } from 'echarts/components';
import { TooltipComponent } from 'echarts/components';


const pick = (obj, keys) => {
  const t = {};
  keys.forEach(key => {
    t[key] = obj[key];
  });
  return t;
};

class EchartsCore extends Component {

  constructor(props) {
    super(props);
    this.echartsLib = echarts;
    this.echartsElement = null;
    this.prepare = props.prepare;
  }

  componentDidMount() {
    this.renderEchartDom();
  }

  componentDidUpdate(prevProps) {
    // 第二次更新时候执行了这个方法
    if (!isEqual(prevProps.theme, this.props.theme)
        || !isEqual(prevProps.opts, this.props.opts)
        || !isEqual(prevProps.onEvents, this.props.onEvents)
    ) {
      // this.dispose(); // 这个方法把之前的元素清掉了
      this.rerender();
      return;
    }
    const pickKeys = ['option', 'notMerge', 'lazyUpdate', 'showLoading', 'loadingOption'];

    if (!isEqual(pick(this.props, pickKeys), pick(prevProps, pickKeys))) {
      return;
    }
    if (typeof this.props.shouldSetOption === 'function' && !this.props.shouldSetOption(prevProps, this.props)) {
      return;
    }
    const echartObj = this.renderEchartDom();
    if (!isEqual(prevProps.style, this.props.style) || !isEqual(prevProps.className, this.props.className)) {
      try {
        if (echartObj) echartObj.resize();
      } catch (e) {
        console.warn(e);
      }
    }
  }

  componentWillUnmount() {
    this.dispose();
  }

  getEchartsInstance = () => this.echartsLib.getInstanceByDom(this.echartsElement) || this.echartsLib.init(this.echartsElement, this.props.theme, this.props.opts)

  renderEchartDom = () => {
    const  now = new Date().getSeconds();
    this.prepare(echarts);
    console.info(" Prepare took: " + (new Date().getSeconds() - now));
    const echartsObj = this.getEchartsInstance();

    // echarts.use([LineChart]);
    // echarts.use([TooltipComponent, GridComponent, LineChart]);

    console.info("option: ");
    console.info(this.props.option);
    // console.info("echarts element: ");
    // console.info(this.echartsElement);
    // console.info("echarts element style: ");
    // console.info(this.echartsElement.style);
    // console.info(this.echartsElement.style.height);
    // console.info(this.echartsElement.style.width);
    // console.info(this.echartsElement.offsetHeight);
    // console.info(this.echartsElement.offsetWidth);

    echartsObj.setOption(this.props.option, this.props.notMerge || false, this.props.lazyUpdate || false);

    const onClickFunc = this.props.onClick;
    if(onClickFunc !== null && onClickFunc !== undefined){
      echartsObj.on('click', onClickFunc);
    }

    window.addEventListener('resize', () => {
      if (echartsObj) echartsObj.resize();
    });
    if (this.props.showLoading) echartsObj.showLoading(this.props.loadingOption || null);
    else echartsObj.hideLoading();
    return echartsObj;
  };

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

  rerender = () => {
    const { onEvents, onChartReady } = this.props;
    const echartsObj = this.renderEchartDom();
    this.bindEvents(echartsObj, onEvents || {});

    if (typeof onChartReady === 'function') this.props.onChartReady(echartsObj);
    // if (this.echartsElement) {
    //   bind( () => {
    //     try {
    //       if (echartsObj) echartsObj.resize();
    //     } catch (e) {
    //       console.warn(e);
    //     }
    //   }, this.echartsElement);
    // }
  };

  bindEvents = (instance, events) => {
    const bindEvent = (eventName, func) => {
      if (typeof eventName === 'string' && typeof func === 'function') {
        instance.on(eventName, param => {
          func(param, instance);
        });
      }
    };
    for (const eventName in events) {
      if (Object.prototype.hasOwnProperty.call(events, eventName)) {
        bindEvent(eventName, events[eventName]);
      }
    }
  };

  render() {
    console.info("echarts render: ");
    const { style, className } = this.props;
    const styleConfig = {
      height: 300,
      ...style,
    };
    const d = <div
        ref={(e) => { this.echartsElement = e; }}
        style={styleConfig}
        className={className}
      />;
    return (d)
  }
}

EchartsCore.propTypes = {
  option: PropTypes.object.isRequired,
  prepare: PropTypes.func,
  style: PropTypes.object,
  className: PropTypes.string,
  notMerge: PropTypes.bool,
  lazyUpdate: PropTypes.bool,
  showLoading: PropTypes.bool,
  loadingOption: PropTypes.object,
  theme: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.object,
  ]),
  onChartReady: PropTypes.func,
  onEvents: PropTypes.object,
  opts: PropTypes.shape({
    devicePixelRatio: PropTypes.number,
    renderer: PropTypes.oneOf(['canvas', 'svg']),
    width: PropTypes.oneOfType([
      PropTypes.number,
      PropTypes.oneOf([null, undefined, 'auto']),
    ]),
    height: PropTypes.oneOfType([
      PropTypes.number,
      PropTypes.oneOf([null, undefined, 'auto']),
    ]),
  }),
  shouldSetOption: PropTypes.func,
};

EchartsCore.defaultProps = {
  style: {},
  className: '',
  prepare: (echarts) => {
    echarts.use([LineChart, GridComponent]);  // this should be above of init echarts
  },
  notMerge: false,
  lazyUpdate: false,
  showLoading: false,
  loadingOption: null,
  theme: null,
  onChartReady: () => {},
  onEvents: {},
  opts: {},
  shouldSetOption: () => true,
};
export default EchartsCore;
