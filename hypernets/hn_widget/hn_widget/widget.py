import json

import ipywidgets as widgets
from traitlets import Unicode
from traitlets import Unicode, Dict
# See js/lib/example.js for the frontend counterpart to this file.
from ipywidgets import widget_serialization
from hn_widget import experiment_util

# See js/lib/example.js for the frontend counterpart to this file.

VIEW_MODULE = "hn_widget"
MODEL_MODULE = "hn_widget"

@widgets.register
class ExperimentSummary(widgets.DOMWidget):

    _view_name = Unicode('ExperimentSummaryView').tag(sync=True)

    _model_name = Unicode('ExperimentSummaryModel').tag(sync=True)

    _view_module = Unicode(VIEW_MODULE).tag(sync=True)

    _model_module = Unicode(MODEL_MODULE).tag(sync=True)

    _view_module_version = Unicode('^0.1.0').tag(sync=True)

    _model_module_version = Unicode('^0.1.0').tag(sync=True)

    value = Dict({}).tag(sync=True, **widget_serialization)

    def __init__(self, compete_experiment, **kwargs):
        super(ExperimentSummary, self).__init__(**kwargs)
        self.value = experiment_util.extract_experiment(compete_experiment)


@widgets.register
class DatasetSummary(widgets.DOMWidget):

    _view_name = Unicode('DatasetSummaryView').tag(sync=True)

    _model_name = Unicode('DatasetSummaryModel').tag(sync=True)

    _view_module = Unicode(VIEW_MODULE).tag(sync=True)

    _model_module = Unicode(MODEL_MODULE).tag(sync=True)

    _view_module_version = Unicode('^0.1.0').tag(sync=True)

    _model_module_version = Unicode('^0.1.0').tag(sync=True)

    value = Dict({}).tag(sync=True, **widget_serialization)

    def __init__(self, data, **kwargs):
        super(DatasetSummary, self).__init__(**kwargs)
        self.value = data


@widgets.register
class ExperimentProcessWidget(widgets.DOMWidget):

    _view_name = Unicode('ExperimentProcessWidgetView').tag(sync=True)

    _model_name = Unicode('ExperimentProcessWidgetModel').tag(sync=True)

    _view_module = Unicode(VIEW_MODULE).tag(sync=True)

    _model_module = Unicode(MODEL_MODULE).tag(sync=True)

    _view_module_version = Unicode('^0.1.0').tag(sync=True)

    _model_module_version = Unicode('^0.1.0').tag(sync=True)

    value = Dict({}).tag(sync=True, **widget_serialization)

    initData = Unicode().tag(sync=True, **widget_serialization)  # use dict will cause  Error setting state: Unexpected token o in JSON at position 1

    def __init__(self, compete_experiment, **kwargs):
        super(ExperimentProcessWidget, self).__init__(**kwargs)
        d = json.dumps(experiment_util.extract_experiment(compete_experiment))
        self.initData = d
