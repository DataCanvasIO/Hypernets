import React, { Component } from 'react';
import * as echarts from "echarts/lib/echarts";
import 'echarts/lib/chart/heatmap';
import PropTypes from 'prop-types';
import { clear } from 'echarts/lib/util/throttle';
import { isEqual } from 'date-fns';
import { bind } from 'zrender/lib/core/util';

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

  }

  componentDidMount() {
    this.renderEchartDom();
  }

  componentDidUpdate(prevProps) {
    if (!isEqual(prevProps.theme, this.props.theme)
        || !isEqual(prevProps.opts, this.props.opts)
        || !isEqual(prevProps.onEvents, this.props.onEvents)
    ) {
      this.dispose();
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
    const echartsObj = this.getEchartsInstance();
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
  }

  rerender = () => {
    const { onEvents, onChartReady } = this.props;
    const echartsObj = this.renderEchartDom();
    this.bindEvents(echartsObj, onEvents || {});
    if (typeof onChartReady === 'function') this.props.onChartReady(echartsObj);
    if (this.echartsElement) {
      bind(this.echartsElement, () => {
        try {
          if (echartsObj) echartsObj.resize();
        } catch (e) {
          console.warn(e);
        }
      });
    }
  }

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
  }

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

EchartsCore.propTypes = {
  option: PropTypes.object.isRequired,
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
