import ipywidgets as widgets
from traitlets import Unicode, Dict, List
from threading import Thread
# See js/lib/example.js for the frontend counterpart to this file.
import time
from ipywidgets import DOMWidget, Layout, widget_serialization
import threading
from  queue import Queue

experiment_event_loacl = threading.local()

@widgets.register
class HelloWorld(widgets.DOMWidget):
    """An example widget."""

    # Name of the widget view class in front-end
    _view_name = Unicode('HelloView').tag(sync=True)

    # Name of the widget model class in front-end
    _model_name = Unicode('HelloModel').tag(sync=True)

    # Name of the front-end module containing widget view
    _view_module = Unicode('jupyter-widget-example').tag(sync=True)

    # Name of the front-end module containing widget model
    _model_module = Unicode('jupyter-widget-example').tag(sync=True)

    # Version of the front-end module containing widget view
    _view_module_version = Unicode('^0.1.0').tag(sync=True)

    # Version of the front-end module containing widget model
    _model_module_version = Unicode('^0.1.0').tag(sync=True)

    # Widget specific property.
    # Widget properties are defined as traitlets. Any property tagged with `sync=True`
    # is automatically synced to the frontend *any* time it changes in Python.
    # It is synced back to Python from the frontend *any* time the model is touched.
    # value = Unicode('Hello World!').tag(sync=True)
    value = Dict({}).tag(sync=True, **widget_serialization)

    initData = Unicode().tag(sync=True, **widget_serialization)


# ipythonwidget.py
    def __init__(self, **kwargs):
        super(HelloWorld, self).__init__(**kwargs)
        def t_main():
            if hasattr(experiment_event_loacl, 'queue') is False:
                experiment_event_loacl.queue = Queue()
            print("queue2 object")
            print(experiment_event_loacl.queue)
            while True:
                event_dict = experiment_event_loacl.queue.get()  # Block and sync queue
                self.value = event_dict
        # Thread(target=t_main).start()
