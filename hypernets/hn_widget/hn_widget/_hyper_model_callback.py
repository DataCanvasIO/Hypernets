import pickle
import time

from hypernets.core.callbacks import Callback, EarlyStoppingCallback
from hypernets.utils.experiment import get_tree_importances, EarlyStoppingConfigMeta, EarlyStoppingStatusMeta
from hypernets.utils import fs

from .mgr import ActionType, send_action


def sort_imp(imp_dict, sort_imp_dict, n_features=10):
    sort_imps = []
    for k in sort_imp_dict:
        sort_imps.append({
            'name': k,
            'imp': sort_imp_dict[k]
        })

    top_features = list(map(lambda x: x['name'], sorted(sort_imps, key=lambda v: v['imp'], reverse=True)[: n_features]))

    imps = []
    for f in top_features:
        imps.append({
            'name': f,
            'imp': imp_dict[f]
        })
    return imps


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
                    if c.start_time is not None:
                        elapsed_time = time.time() - c.start_time
                    else:
                        elapsed_time = None
                    ess = EarlyStoppingStatusMeta(c.best_reward, c.best_trial_no, c.counter_no_improvement_trials, c.triggered, c.triggered_reason, elapsed_time)
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
                imps.append(get_tree_importances(m))

            imps_avg = {}
            for k in imps[0]:
                imps_avg[k] = sum([imp.get(k, 0) for imp in imps]) / 3

            for fold, m in enumerate(cv_models):
                models_json.append({
                    'fold': fold,
                    'importances': sort_imp(get_tree_importances(m), imps_avg)
                })
        else:
            gbm_model = model.gbm_model
            if gbm_model is None:
                raise Exception("Both cv_models or gbm_model is None ")
            imp_dict = get_tree_importances(gbm_model)
            models_json.append({
                'fold': None,
                'importances': sort_imp(imp_dict, imp_dict)
            })
        early_stopping_status = None
        early_stopping_config = None
        for c in hyper_model.callbacks:
            if isinstance(c, EarlyStoppingCallback):
                early_stopping_status = EarlyStoppingStatusMeta(c.best_reward, c.best_trial_no, c.counter_no_improvement_trials, c.triggered, c.triggered_reason, time.time() - c.start_time).to_dict()
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
