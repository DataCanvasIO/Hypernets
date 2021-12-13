import time

from IPython.display import display

from hypernets.experiment import ExperimentCallback
from hypernets.utils.experiment import ExperimentExtractor, StepMeta

from .widget import ExperimentProcessWidget
from ._hyper_model_callback import JupyterHyperModelCallback
from .mgr import ActionType, send_action, set_widget

MAX_IMPORTANCE_NUM = 10  # TOP N important features


class JupyterWidgetExperimentCallback(ExperimentCallback):

    def __init__(self):
        self.widget_id = id(self)

    @staticmethod
    def get_step_index(experiment, step_name):
        for i, step in enumerate(experiment.steps):
            if step.name == step_name:
                return i
        return -1

    @staticmethod
    def set_up_hyper_model_callback(exp, handler):
        for c in exp.hyper_model.callbacks:
            if isinstance(c, JupyterHyperModelCallback):
                handler(c)
                break

    def experiment_start(self, exp):
        self.set_up_hyper_model_callback(exp, lambda c: c.set_widget_id(self.widget_id))
        # c.set_step_index(i)
        dom_widget = ExperimentProcessWidget(exp)
        set_widget(self.widget_id, dom_widget)
        display(dom_widget)
        dom_widget.initData = ''  # remove init data, if refresh the page will show nothing on the browser

    def experiment_end(self, exp, elapsed):
        send_action(self.widget_id, ActionType.ExperimentFinish, {})

    def experiment_break(self, exp, error):
        send_action(self.widget_id, ActionType.ExperimentBreak, {})

    def step_start(self, exp, step):
        step_name = step
        step_index = self.get_step_index(exp, step_name)
        self.set_up_hyper_model_callback(exp, lambda c: c.set_step_index(step_index))
        payload = {
            'index': step_index,
            'status': StepMeta.STATUS_PROCESS,
            'start_datetime': time.time()
        }
        send_action(self.widget_id, ActionType.StepBegin, payload)

    def step_progress(self, exp, step, progress, elapsed, eta=None):
        pass

    def step_end(self, exp, step, output, elapsed):
        step_name = step
        step = exp.get_step(step_name)
        # setattr(step, 'status', StepStatus.Finish)
        # todo set time setattr(step, 'status', StepStatus.Finish)
        step.done_time = time.time()  # fix done_time is none
        step_index = self.get_step_index(exp, step_name)
        d = ExperimentExtractor.extract_step(step_index, step).to_dict()
        send_action(self.widget_id, ActionType.StepFinished, d)

    def step_break(self, exp, step, error):
        step_name = step
        step_index = self.get_step_index(exp, step_name)
        self.set_up_hyper_model_callback(exp, lambda c: c.set_step_index(step_index))
        payload = {
            'index': step_index,
            'extension': {
                'reason': str(error)
            },
            'status': StepMeta.STATUS_ERROR
        }
        send_action(self.widget_id, ActionType.StepBegin, payload)
