var widgets = require('@jupyter-widgets/base');

var pipelineVis = require('./pipelineVis.js');

var _ = require('lodash');

var HelloModel = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _model_name : 'HelloModel',
        _view_name : 'HelloView',
        _model_module : 'jupyter-widget-example',
        _view_module : 'jupyter-widget-example',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
        value : 'Hello World!'
    })
});


// Custom View. Renders the widget model.
var HelloView = widgets.DOMWidgetView.extend({
    // Defines how the widget gets rendered into the DOM
    render: function() {
        // this.value_changed();
        console.log('pipelineVis lib: ');
        console.log(pipelineVis);
        console.log(pipelineVis.pipelineVis);
        console.log(pipelineVis.pipelineVis.store);
        //

        // Observe changes in the value traitlet in Python, and define
        // a custom callback.
        this.model.on('change:value', this.value_changed, this);
        this.model.on('change:initData', this.init_data_changed, this);
    },

    init_data_changed: function(){
        const initData = JSON.parse(this.model.get('initData'));
        // todo initData convent to JSON
        console.log("Received init_data_changed from backend: ");
        console.info(JSON.stringify(initData));
        pipelineVis.pipelineVis.renderPipelineMatrixBundle(this.el, initData);
    },

    value_changed: function() {
        // this.el.textContent = this.model.get('value');
        const value = this.model.get('value');
        console.log("Received value_changed from backend: ");
        console.log(JSON.stringify(value));
        pipelineVis.pipelineVis.store.dispatch({ type: 'stepFinished' , data: value });
    }
});

module.exports = {
    HelloModel: HelloModel,
    HelloView: HelloView
};
