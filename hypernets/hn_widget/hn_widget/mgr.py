_DOM_WIDGETS = {}


class ActionType:
    EarlyStopped = 'earlyStopped'
    StepFinished = 'stepFinished'
    StepBegin = 'stepBegin'
    StepError = 'stepError'
    TrialFinished = 'trialFinished'
    ExperimentFinish = 'experimentFinish'
    ExperimentBreak = 'experimentBreak'


def set_widget(widget_id, widget_obj):
    global _DOM_WIDGETS
    _DOM_WIDGETS[widget_id] = widget_obj


def send_action(widget_id, action_type, data):
    global _DOM_WIDGETS
    dom_widget = _DOM_WIDGETS.get(widget_id)
    if dom_widget is None:
        raise ValueError(f"widget_id: {widget_id} not exists ")
    action = {'type': action_type, 'payload': data}
    # print("----action-----")
    # print(action)
    dom_widget.value = action
