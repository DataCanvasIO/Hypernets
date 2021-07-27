var widgets = require('@jupyter-widgets/base');
var _ = require('lodash');

// var hypernetsExperiment = require('./hypernetsExperiment.js');
var hypernetsExperiment = require('hypernets-experiment').hypernetsExperiment;

var ExperimentSummaryModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'ExperimentSummaryModel',
        _view_name : 'ExperimentSummaryView',
        _model_module : 'hn_widget',
        _view_module : 'hn_widget',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
        value : ''
    })
});


// Custom View. Renders the widget model.
var ExperimentSummaryView = widgets.DOMWidgetView.extend({
    // Defines how the widget gets rendered into the DOM
    render: function() {
        // this.value_changed();
        console.log('hypernetsExperiment lib in renderExperimentSummary: ');
        console.log(hypernetsExperiment);
        const value = this.model.get('value');
        hypernetsExperiment.renderExperimentSummary(value, this.el);
    }
});

var DatasetSummaryModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'DatasetSummaryModel',
        _view_name : 'DatasetSummaryView',
        _model_module : 'hn_widget',
        _view_module : 'hn_widget',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
        value : 'Hello World!'
    })
});


// Custom View. Renders the widget model.
var DatasetSummaryView = widgets.DOMWidgetView.extend({
    // Defines how the widget gets rendered into the DOM
    render: function() {
        // this.value_changed();
        console.log('hypernetsExperiment lib in renderDatasetSummary updated2: ');
        console.log(hypernetsExperiment);
        console.log(this.el);
        const value = this.model.get('value');
        hypernetsExperiment.renderDatasetSummary(value, this.el);
    }
});


var ExperimentProcessWidgetModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'ExperimentProcessWidgetModel',
        _view_name : 'ExperimentProcessWidgetView',
        _model_module : 'hn_widget',
        _view_module : 'hn_widget',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
        value : {},
        initData : '',
    })
});

var ExperimentProcessWidgetView = widgets.DOMWidgetView.extend({
    reactStore: null,
    // Defines how the widget gets rendered into the DOM
    render: function() {
        console.log('hypernetsExperiment lib 1 : ');
        console.log(hypernetsExperiment);

        console.info("Received origin init data from backend1111: ");
        const originInitData = this.model.get('initData');
        console.info("Received origin init data from backend: ");
        console.info(originInitData);

        if(originInitData !== null && originInitData !== undefined  && originInitData.length > 0){
            const initData = JSON.parse(originInitData);
            const steps = initData['steps'];
            if (steps !== null && steps !== undefined && steps.length > 0){
                this.reactStore = hypernetsExperiment.renderExperimentProcess(initData, this.el);
                console.info("created store: ");
                console.info(this.reactStore);
                this.model.on('change:value', this.value_changed, this);
                // this.model.on('change:initData', this.init_data_changed, this);
            }else{
                console.warn("Received steps in init_data is empty ");
            }
        }else{
            hypernetsExperiment.renderLossState(this.el);
            console.error("Received init_data is null ");
        }
    },

    value_changed: function() {
        // this.el.textContent = this.model.get('value');
        const value = this.model.get('value');
        //        console.log("Received value_changed from backend: ");  // print this value already in react project
        //        console.log(value);
        if (this.reactStore !== null && this.reactStore !== undefined){
            // if is experiment end or experiment break, sync the state data to backend
            const actionType = value.type;
            if(actionType === 'experimentFinish' || actionType === 'experimentBreak' ){
                console.log("Sync the state data to backend ");
                this.model.set('initData', JSON.stringify(this.reactStore.getState()));
                this.model.save_changes();
            }else{
                // do not handle experimentFinish,experimentBreak action type in react project
                this.reactStore.dispatch(value);
            }
        }else{
            console.warn("state store is null, please check is the widget initialize succeed");
        }
    }
});


module.exports = {
    ExperimentSummaryModel: ExperimentSummaryModel,
    ExperimentSummaryView: ExperimentSummaryView,
    DatasetSummaryModel: DatasetSummaryModel,
    DatasetSummaryView: DatasetSummaryView,
    ExperimentProcessWidgetModel: ExperimentProcessWidgetModel,
    ExperimentProcessWidgetView: ExperimentProcessWidgetView
};
