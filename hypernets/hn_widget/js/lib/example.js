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
        value : 'Hello World!'
    })
});


// Custom View. Renders the widget model.
var ExperimentProcessWidgetView = widgets.DOMWidgetView.extend({
    reactStore: null,
    // Defines how the widget gets rendered into the DOM
    render: function() {
        // this.value_changed();
        console.log('hypernetsExperiment lib: ');
        console.log(hypernetsExperiment);

        const originInitData = this.model.get('initData');
        if(originInitData !== null && originInitData !== undefined){
            const initData = JSON.parse(originInitData);
            console.log("Received init_data_changed from backend");
            console.info(initData);
            this.reactStore = hypernetsExperiment.renderExperimentProcess(initData, this.el);
            console.info("store: ");
            console.info(this.reactStore);
            this.model.on('change:value', this.value_changed, this);
            // this.model.on('change:initData', this.init_data_changed, this);
        }else{
            console.warn("Received init_data_changed is null");
        }
    },

    value_changed: function() {
        // this.el.textContent = this.model.get('value');
        const value = this.model.get('value');
        console.log("Received value_changed from backend: ");
        console.log(JSON.stringify(value));
        this.reactStore.dispatch(value);

        // if is experiment end or experiment break, sync the state data to backend
        const actionType = value.type;
        if(actionType === 'experimentFinish' || actionType === 'experimentBreak' ){
            this.model.set('initData', this.reactStore.getState());
            this.model.save_changes();
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
