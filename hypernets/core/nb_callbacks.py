
# hk.search(X_train, y_train, X_test, y_test, cv=False, max_trials=3)
from hn_widget.experiment_util import StepStatus

from hypernets.experiment import ExperimentCallback
from hypernets.core.callbacks import Callback
import json
from IPython.display import display_html, HTML, display
import pickle

from hypernets.experiment.compete import SpaceSearchStep
from hypernets.utils import fs
from hypernets.core.callbacks import EarlyStoppingCallback
import time
import lightgbm as lgb
import xgboost as xgb
import catboost
from xgboost.sklearn import XGBModel
from lightgbm.sklearn import LGBMModel
from catboost.core import CatBoost

from hn_widget.widget import ExperimentProcessWidget
from hn_widget.experiment_util import EarlyStoppingStatus, EarlyStoppingConfig

MAX_IMPORTANCE_NUM = 10


def extract_importances(gbm_model):

    def get_imp(n_features):
        try:
            return gbm_model.feature_importances_
        except Exception as e:
            # print(e)
            return [0 for i in range(n_features)]

    if isinstance(gbm_model, XGBModel):
        importances_pairs = list(zip(gbm_model._Booster.feature_names, get_imp(len(gbm_model._Booster.feature_names))))
    elif isinstance(gbm_model, LGBMModel):
        if hasattr(gbm_model, 'feature_name_'):
            names = gbm_model.feature_name_
        else:
            names = [f'col_{i}' for i in range(gbm_model.feature_importances_.shape[0])]
        importances_pairs = list(zip(names, get_imp(len(names))))
    elif isinstance(gbm_model, CatBoost):
        importances_pairs = list(zip(gbm_model.feature_names_, get_imp(len(gbm_model.feature_names_))))
    else:
        importances_pairs = []

    importances = {}
    for name, imp in importances_pairs:
        importances[name] = imp

    return importances


def sort_imp(imp_dict, sort_imp_dict):
    sort_imps = []
    for k in sort_imp_dict:
        sort_imps.append({
            'name': k,
            'imp': sort_imp_dict[k]
        })

    top_features = list(map(lambda x: x['name'], sorted(sort_imps, key=lambda v: v['imp'], reverse=True)[: MAX_IMPORTANCE_NUM]))

    imps = []
    for f in top_features:
        imps.append({
            'name': f,
            'imp': imp_dict[f]
        })
    return imps


def send_action(widget_id, action_type, data):
    dom_widget = DOM_WIDGETS.get(widget_id)
    if dom_widget is None:
        raise Exception(f"widget_id: {widget_id} not exists ")
    action = {'type': action_type, 'payload': data}
    # print("----action-----")
    # print(action)
    dom_widget.value = action


class ActionType:
    EarlyStopped = 'earlyStopped'
    StepFinished = 'stepFinished'
    StepBegin = 'stepBegin'
    StepError = 'stepError'
    TrialFinished = 'trialFinished'
    ExperimentFinish = 'experimentFinish'
    ExperimentBreak = 'experimentBreak'


class JupyterHyperModelCallback(Callback):

    def __init__(self):
        super(JupyterHyperModelCallback, self).__init__()
        self.widget_id = None
        self.step_index = None
        self.max_trials = None

    def set_widget_id(self, widget_id):
        self.widget_id = widget_id

    def set_step_index(self, value):
        self.step_index = value

    def on_search_start(self, hyper_model, X, y, X_eval, y_eval, cv, num_folds, max_trials, dataset_id, trial_store,
                        **fit_kwargs):
        self.max_trials = max_trials

    def on_search_end(self, hyper_model):
        for c in hyper_model.callbacks:
            if isinstance(c, EarlyStoppingCallback):
                if c.triggered:
                    # if c.triggered_reason == EarlyStoppingCallback.REASON_TIME_LIMIT:
                    #     value = c.time_limit
                    # elif c.triggered_reason == EarlyStoppingCallback.REASON_TRIAL_LIMIT:
                    #     value = c.counter_no_improvement_trials
                    # elif c.triggered_reason == EarlyStoppingCallback.REASON_EXPECTED_REWARD:
                    #     value = c.best_reward
                    # else:
                    #     raise Exception("Unseen reason " + c.triggered_reason)
                    if c.start_time is not None:
                        elapsed_time = time.time() - c.start_time
                    else:
                        elapsed_time = None
                    ess = EarlyStoppingStatus(c.best_reward, c.best_trial_no, c.counter_no_improvement_trials, c.triggered, c.triggered_reason, elapsed_time)
                    payload = {
                        'stepIndex': self.step_index,
                        'data': ess.to_dict()
                    }
                    send_action(self.widget_id, ActionType.EarlyStopped, payload)

    def on_search_error(self, hyper_model):
        pass

    def on_build_estimator(self, hyper_model, space, estimator, trial_no):
        pass

    def on_trial_begin(self, hyper_model, space, trial_no):
        pass

    @staticmethod
    def get_space_params(space):
        params_dict = {}
        for hyper_param in space.get_assigned_params():
            # param_name = hyper_param.alias[len(list(hyper_param.references)[0].name) + 1:]
            param_name = hyper_param.alias
            param_value = hyper_param.value
            # only show number param
            # if isinstance(param_value, int) or isinstance(param_value, float):
            #     if not isinstance(param_value, bool):
            #         params_dict[param_name] = param_value
            if param_name is not None and param_value is not None:
                # params_dict[param_name.split('.')[-1]] = str(param_value)
                params_dict[param_name] = str(param_value)
        return params_dict

    def ensure_number(self, value, var_name):
        if value is None:
             raise ValueError(f"Var {var_name} can not be None.")
        else:
            if not isinstance(value, float) and not isinstance(value, int):
                raise ValueError(f"Var {var_name} = {value} not a number.")

    def on_trial_end(self, hyper_model, space, trial_no, reward, improved, elapsed):
        self.ensure_number(reward, 'reward')
        self.ensure_number(trial_no, 'trail_no')
        self.ensure_number(elapsed, 'elapsed')
        trial = None
        for t in hyper_model.history.trials:
            if t.trial_no == trial_no:
                trial = t
                break

        if trial is None:
            raise Exception(f"Trial no {trial_no} is not in history")

        model_file = trial.model_file
        with fs.open(model_file, 'rb') as input:
            model = pickle.load(input)

        cv_models = model.cv_gbm_models_
        models_json = []
        is_cv = cv_models is not None and len(cv_models) > 0
        if is_cv:
            # cv is opening
            imps = []
            for m in cv_models:
                imps.append(extract_importances(m))

            imps_avg = {}
            for k in imps[0]:
                imps_avg[k] = sum([imp.get(k, 0) for imp in imps]) / 3

            for fold, m in enumerate(cv_models):
                models_json.append({
                    'fold': fold,
                    'importances': sort_imp(extract_importances(m), imps_avg)
                })
        else:
            gbm_model = model.gbm_model
            if gbm_model is None:
                raise Exception("Both cv_models or gbm_model is None ")
            imp_dict = extract_importances(gbm_model)
            models_json.append({
                'fold': None,
                'importances': sort_imp(imp_dict, imp_dict)
            })
        early_stopping_status = None
        early_stopping_config = None
        for c in hyper_model.callbacks:
            if isinstance(c, EarlyStoppingCallback):
                early_stopping_status = EarlyStoppingStatus(c.best_reward, c.best_trial_no, c.counter_no_improvement_trials, c.triggered, c.triggered_reason, time.time() - c.start_time).to_dict()
                break
        data = {
            'stepIndex': self.step_index,
            'data': {
                "trialNo": trial_no,
                "maxTrials": self.max_trials,
                "hyperParams": self.get_space_params(space),
                "models": models_json,
                "reward": reward,
                "elapsed": elapsed,
                "is_cv": is_cv,
                "metricName": hyper_model.reward_metric,
                "earlyStopping": early_stopping_status
            }
        }
        send_action(self.widget_id, ActionType.TrialFinished, data)

    def on_trial_error(self, hyper_model, space, trial_no):
        pass

    def on_skip_trial(self, hyper_model, space, trial_no, reason, reward, improved, elapsed):
        pass


DOM_WIDGETS = {}


class JupyterWidgetExperimentCallback(ExperimentCallback):

    def __init__(self):
        self.widget_id = id(self)

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
        DOM_WIDGETS[self.widget_id] = dom_widget
        display(dom_widget)
        dom_widget.initData = ''  # remove init data, if refresh the page will show nothing on the browser

    def experiment_end(self, exp, elapsed):
        send_action(self.widget_id, ActionType.ExperimentFinish, {})

    def experiment_break(self, exp, error):
        send_action(self.widget_id, ActionType.ExperimentBreak, {})

    def step_start(self, exp, step):
        from hn_widget.experiment_util import get_step_index
        from hn_widget.experiment_util import StepStatus
        step_name = step
        step_index = get_step_index(exp, step_name)
        self.set_up_hyper_model_callback(exp, lambda c: c.set_step_index(step_index))
        payload = {
            'index': step_index,
            'status': StepStatus.Process,
            'start_datetime': time.time()
        }
        send_action(self.widget_id, ActionType.StepBegin, payload)

    def step_progress(self, exp, step, progress, elapsed, eta=None):
        pass

    def step_end(self, exp, step, output, elapsed):
        from hn_widget import experiment_util
        from hn_widget.experiment_util import StepStatus
        step_name = step
        step = exp.get_step(step_name)
        # setattr(step, 'status', StepStatus.Finish)
        # todo set time setattr(step, 'status', StepStatus.Finish)
        step.done_time = time.time()  # fix done_time is none
        step_index = experiment_util.get_step_index(exp, step_name)
        d = experiment_util.extract_step(step_index, step)
        send_action(self.widget_id, ActionType.StepFinished, d)

    def step_break(self, exp, step, error):
        from hn_widget.experiment_util import get_step_index
        from hn_widget.experiment_util import StepStatus
        step_name = step
        step_index = get_step_index(exp, step_name)
        self.set_up_hyper_model_callback(exp, lambda c: c.set_step_index(step_index))
        payload = {
            'index': step_index,
            'extension': {
                'reason': str(error)
            },
            'status': StepStatus.Error
        }
        send_action(self.widget_id, ActionType.StepBegin, payload)
