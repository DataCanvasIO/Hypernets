
# hk.search(X_train, y_train, X_test, y_test, cv=False, max_trials=3)
from hypernets.experiment import ExperimentCallback
from hypernets.core.callbacks import Callback
import json
from IPython.display import display_html, HTML, display
import pickle
from hypernets.utils import fs
from hypernets.core.callbacks import EarlyStoppingCallback
import time


class JupyterHyperModelCallback(Callback):

    def __init__(self):
        super(JupyterHyperModelCallback, self).__init__()
        self.dom_widget = None

    def set_dom_widget(self, dom_widget):
        self.dom_widget = dom_widget

    def on_search_start(self, hyper_model, X, y, X_eval, y_eval, cv, num_folds, max_trials, dataset_id, trial_store,
                        **fit_kwargs):
        pass

    def on_search_end(self, hyper_model):
        pass

    def on_search_error(self, hyper_model):
        pass

    def on_build_estimator(self, hyper_model, space, estimator, trial_no):
        pass

    def on_trial_begin(self, hyper_model, space, trial_no):
        pass

    @staticmethod
    def get_space_params(space):
        params_dict = {}
        for hyper_param in space.get_all_params():
            # param_name = hyper_param.alias[len(list(hyper_param.references)[0].name) + 1:]
            param_name = hyper_param.alias
            param_value = hyper_param.value
            # only show number param
            # if isinstance(param_value, int) or isinstance(param_value, float):
            #     if not isinstance(param_value, bool):
            #         params_dict[param_name] = param_value
            params_dict[param_name] = param_value
        return params_dict

    def ensure_number(self, value, var_name):
        if value is None:
             raise ValueError(f"Var {var_name} can not be None.")
        else:
            if not isinstance(value, float) and not isinstance(value, int):
                raise ValueError(f"Var {var_name} = {value} not a number.")

    def on_trial_end(self, hyper_model, space, trial_no, reward, improved, elapsed):
        # 1. 获取超参数
        # 2. 获取模型重要性信息，还有reward
        # 3. 还有指标的名称 #
        # 4. 然后就是earlystopping 的信息
        # 5.

        print(hyper_model)
        print(hyper_model)
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
            print(model)
            print(model)

        cv_models = model.cv_gbm_models_
        models_json = []
        is_cv = cv_models is not None and len(cv_models) > 0
        if is_cv :
            # cv is opening
            for fold, m in enumerate(cv_models):
                models_json.append({
                    'fold': fold,
                    'importances': m.feature_importances_.tolist()
                })
        else:
            gbm_model = model.gbm_model
            if gbm_model is None:
                raise Exception("Both cv_models or gbm_model is None ")
            models_json.append({
                'fold': None,
                'importances': gbm_model.feature_importances_.tolist()
            })

        earlyStopping_status = {
            'reward': None,
            'noImprovedTrials': None,
            'elapsedTime': None
        }

        earlyStoppingCallback = None
        for c in hyper_model.callbacks:
            if isinstance(c, EarlyStoppingCallback):
                earlyStopping_status = {
                    'reward': hyper_model.best_reward,
                    'noImprovedTrials': c.counter_no_improvement_trials + 2,
                    'elapsedTime': time.time() - c.start_time
                }
                break

        return {
            "trialNo": trial_no,
            "hyperParams": self.get_space_params(space),
            "models": models_json,
            "reward": reward,
            "elapsed": elapsed,
            "is_cv": is_cv,
            "metricName": hyper_model.reward_metric,
            "earlyStoppingStatus": earlyStopping_status
        }

    def on_trial_error(self, hyper_model, space, trial_no):

        pass

    def on_skip_trial(self, hyper_model, space, trial_no, reason, reward, improved, elapsed):
        pass


class JupyterWidgetExperimentCallback(ExperimentCallback):

    def __init__(self):
        from hn_widget.widget import ExperimentProcessWidget
        self.dom_widget = ExperimentProcessWidget()

    def experiment_start(self, exp):
        display(self.dom_widget)
        from hn_widget.experiment_util import extract_experiment
        self.dom_widget.initData = json.dumps(extract_experiment(exp))

    def experiment_end(self, exp, elapsed):
        pass

    def experiment_break(self, exp, error):
        pass

    def step_start(self, exp, step):
        pass

    def step_progress(self, exp, step, progress, elapsed, eta=None):
        pass

    def step_end(self, exp, step, output, elapsed):
        from hn_widget import experiment_util
        from hn_widget.experiment_util import StepStatus
        step_name = step

        step = exp.get_step(step_name)
        setattr(step, 'status', StepStatus.Finish)
        # todo set time setattr(step, 'status', StepStatus.Finish)

        step_index = experiment_util.get_step_index(exp, step_name)
        experiment_util.extract_step(step_index, step)
        self.dom_widget.value = experiment_util.extract_step(step_index, step)

    def step_break(self, exp, step, error):
        pass
