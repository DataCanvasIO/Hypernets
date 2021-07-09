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

        // Observe changes in the value traitlet in Python, and define
        // a custom callback.
        this.model.on('change:value', this.value_changed, this);
        this.model.on('change:initData', this.init_data_changed, this);
    },

    init_data_changed: function(){
        const initData = JSON.parse(this.model.get('initData'));
        console.log("Received init_data_changed from backend");
        console.info(JSON.stringify(initData));
        this.reactStore = hypernetsExperiment.renderExperimentProcess(initData, this.el);
        console.info("store: ");
        console.info(this.reactStore);
    },

    value_changed: function() {
        // this.el.textContent = this.model.get('value');
        const value = this.model.get('value');
        console.log("Received value_changed from backend: ");
        console.log(JSON.stringify(value));
        this.reactStore.dispatch(value);
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
