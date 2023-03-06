# -*- coding:utf-8 -*-
__author__ = 'yangjian'

from ..core import Callback, EarlyStoppingCallback

"""

"""
import abc
import time
import os
from datetime import datetime
from collections import OrderedDict
from threading import Thread, RLock
import math
from typing import Type

import pandas as pd
import numpy as np
import joblib
import psutil
from sklearn import metrics as sk_metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from IPython.display import display, display_markdown

from hypernets.tabular import get_tool_box
from hypernets.utils import const, df_utils
from hypernets.experiment import ExperimentExtractor
from . import ExperimentCallback
from .compete import StepNames
from .report import ReportRender
from ._extractor import ConfusionMatrixMeta, StepMeta, EarlyStoppingStatusMeta


class ConsoleCallback(ExperimentCallback):
    def experiment_start(self, exp):
        print('experiment start')

    def experiment_end(self, exp, elapsed):
        print(f'experiment end')
        print(f'   elapsed:{elapsed}')

    def experiment_break(self, exp, error):
        print(f'experiment break, error:{error}')

    def step_start(self, exp, step):
        print(f'   step start, step:{step}')

    def step_progress(self, exp, step, progress, elapsed, eta=None):
        print(f'      progress:{progress}')
        print(f'         elapsed:{elapsed}')

    def step_end(self, exp, step, output, elapsed):
        print(f'   step end, step:{step}, output:{output.items() if output is not None else ""}')
        print(f'      elapsed:{elapsed}')

    def step_break(self, exp, step, error):
        print(f'step break, step:{step}, error:{error}')


class StepCallback:
    def step_start(self, exp, step):
        pass

    def step_progress(self, exp, step, progress, elapsed, eta=None):
        pass

    def step_end(self, exp, step, output, elapsed):
        pass

    def step_break(self, exp, step, error):
        pass


class NotebookStepCallback(StepCallback):
    def step_start(self, exp, step):
        title = step.replace('_', ' ').title()
        display_markdown(f'## {title}', raw=True)

        params = exp.get_step(step).get_params()
        data = self.params_to_display(params)
        if data is not None:
            display_markdown('### Initliazed parameters', raw=True)
            display(data, display_id=f'input_{step}')

    def step_end(self, exp, step, output, elapsed):
        step_obj = exp.get_step(step)
        fitted_params = step_obj.get_fitted_params()
        data = self.fitted_params_to_display(fitted_params)
        if data is not None:
            display_markdown('### Fitted parameters', raw=True)
            display(data, display_id=f'output_{step}')

        display_markdown('### Elapsed', raw=True)
        display_markdown(f'* {step_obj.elapsed_seconds:.3f} seconds', raw=True)

    @staticmethod
    def params_to_display(params):
        df = pd.DataFrame(params.items(), columns=['key', 'value'])
        return df

    @staticmethod
    def fitted_params_to_display(fitted_params):
        df = pd.DataFrame(fitted_params.items(), columns=['key', 'value'])
        return df


class NotebookFeatureSelectionStepCallback(NotebookStepCallback):
    @staticmethod
    def fitted_params_to_display(fitted_params):
        params = fitted_params.copy()

        selected = params.get('selected_features')
        unselected = params.get('unselected_features')
        if selected is not None and unselected is not None:
            params['kept/dropped feature count'] = f'{len(selected)}/{len(unselected)}'

        return NotebookStepCallback.fitted_params_to_display(params)


class NotebookFeatureImportanceSelectionStepCallback(NotebookStepCallback):

    def step_end(self, exp, step, output, elapsed):
        super().step_end(exp, step, output, elapsed)

        fitted_params = exp.get_step(step).get_fitted_params()
        input_features = fitted_params.get('input_features')
        selected = fitted_params.get('selected_features')
        if selected is None:
            selected = input_features
        importances = fitted_params.get('importances', [])
        is_selected = [input_features[i] in selected for i in range(len(importances))]
        df = pd.DataFrame(
            zip(input_features, importances, is_selected),
            columns=['feature', 'importance', 'selected'])
        df = df.sort_values('importance', axis=0, ascending=False)

        display_markdown('### Feature importances', raw=True)
        display(df, display_id=f'output_{step}_importances')


class NotebookPermutationImportanceSelectionStepCallback(NotebookStepCallback):

    def step_end(self, exp, step, output, elapsed):
        super().step_end(exp, step, output, elapsed)

        fitted_params = exp.get_step(step).get_fitted_params()
        importances = fitted_params.get('importances')
        if importances is not None:
            df = pd.DataFrame(
                zip(importances['columns'], importances['importances_mean'], importances['importances_std']),
                columns=['feature', 'importance', 'std'])

            display_markdown('### Permutation importances', raw=True)
            display(df, display_id=f'output_{step}_importances')


class NotebookEstimatorBuilderStepCallback(NotebookStepCallback):
    @staticmethod
    def fitted_params_to_display(fitted_params):
        return fitted_params.get('estimator')


class NotebookPseudoLabelingStepCallback(NotebookStepCallback):
    def step_end(self, exp, step, output, elapsed):
        super().step_end(exp, step, output, elapsed)

        try:
            fitted_params = exp.get_step(step).get_fitted_params()
            proba = fitted_params.get('test_proba')
            proba = proba[:, 1]  # fixme for multi-classes

            from scipy.stats import gaussian_kde
            import matplotlib.pyplot as plt

            kde = gaussian_kde(proba)
            kde.set_bandwidth(0.01 * kde.factor)
            x = np.linspace(proba.min(), proba.max(), num=100)
            y = kde(x)

            # Draw Plot
            plt.figure(figsize=(8, 4), dpi=80)
            plt.plot(x, y, 'g-', alpha=0.7)
            plt.fill_between(x, y, color='g', alpha=0.2)
            # Decoration
            plt.title('Density Plot of Probability', fontsize=22)
            plt.show()
        except:
            pass


_default_step_callback = NotebookStepCallback

_step_callbacks = {
    StepNames.DATA_CLEAN: NotebookFeatureSelectionStepCallback,
    StepNames.FEATURE_GENERATION: NotebookStepCallback,
    StepNames.MULITICOLLINEARITY_DETECTION: NotebookFeatureSelectionStepCallback,
    StepNames.DRIFT_DETECTION: NotebookFeatureSelectionStepCallback,
    StepNames.FEATURE_IMPORTANCE_SELECTION: NotebookFeatureImportanceSelectionStepCallback,
    StepNames.SPACE_SEARCHING: NotebookStepCallback,
    StepNames.ENSEMBLE: NotebookEstimatorBuilderStepCallback,
    StepNames.TRAINING: NotebookEstimatorBuilderStepCallback,
    StepNames.PSEUDO_LABELING: NotebookPseudoLabelingStepCallback,
    StepNames.FEATURE_RESELECTION: NotebookPermutationImportanceSelectionStepCallback,
    StepNames.FINAL_SEARCHING: NotebookStepCallback,
    StepNames.FINAL_ENSEMBLE: NotebookEstimatorBuilderStepCallback,
    StepNames.FINAL_TRAINING: NotebookEstimatorBuilderStepCallback,
}


class SimpleNotebookCallback(ExperimentCallback):
    def __init__(self):
        super(SimpleNotebookCallback, self).__init__()

        self.exp = None
        self.steps = None
        self.running = None

    def experiment_start(self, exp):
        self.exp = exp
        self.steps = OrderedDict()
        self.running = True

        display_markdown('### Input Data', raw=True)

        X_train, y_train, X_test, X_eval, y_eval = \
            exp.X_train, exp.y_train, exp.X_test, exp.X_eval, exp.y_eval
        tb = get_tool_box(X_train, y_train, X_test, X_eval, y_eval)
        display_data = (tb.get_shape(X_train),
                        tb.get_shape(y_train),
                        tb.get_shape(X_eval, allow_none=True),
                        tb.get_shape(y_eval, allow_none=True),
                        tb.get_shape(X_test, allow_none=True),
                        exp.task if exp.task == const.TASK_REGRESSION
                        else f'{exp.task}({tb.to_local(y_train.nunique())[0]})')
        display(pd.DataFrame([display_data],
                             columns=['X_train.shape',
                                      'y_train.shape',
                                      'X_eval.shape',
                                      'y_eval.shape',
                                      'X_test.shape',
                                      'Task', ]), display_id='output_intput')

        try:
            import matplotlib.pyplot as plt

            y_train = y_train.dropna()
            if exp.task == const.TASK_REGRESSION:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(y_train)
                kde.set_bandwidth(0.01 * kde.factor)
                x = np.linspace(y_train.min(), y_train.max(), num=100)
                y = kde(x)
                # Draw Plot
                plt.figure(figsize=(8, 4), dpi=80)
                plt.plot(x, y, 'g-', alpha=0.7)
                plt.fill_between(x, y, color='g', alpha=0.2)
            else:
                tb = get_tool_box(y_train)
                vs = tb.value_counts(y_train)
                labels = list(sorted(vs.keys()))
                values = [vs[k] for k in labels]
                # Draw Plot
                plt.figure(figsize=(8, 4), dpi=80)
                plt.pie(values, labels=labels, autopct='%1.1f%%')

            plt.title('Distribution of y', fontsize=22)
            plt.show()
        except:
            pass

    def experiment_end(self, exp, elapsed):
        self.running = False

    def experiment_break(self, exp, error):
        self.running = False

    def step_start(self, exp, step):
        cb_cls = _step_callbacks[step] if step in _step_callbacks.keys() else _default_step_callback
        cb = cb_cls()
        cb.step_start(exp, step)
        self.steps[step] = cb

    def step_progress(self, exp, step, progress, elapsed, eta=None):
        assert self.steps is not None and step in self.steps.keys()
        cb = self.steps[step]
        cb.step_progress(exp, step, progress, elapsed, eta=eta)

    def step_end(self, exp, step, output, elapsed):
        assert self.steps is not None and step in self.steps.keys()
        cb = self.steps[step]
        cb.step_end(exp, step, output, elapsed)

    def step_break(self, exp, step, error):
        assert self.steps is not None and step in self.steps.keys()
        cb = self.steps[step]
        cb.step_break(exp, step, error)


class MLEvaluateCallback(ExperimentCallback):

    def __init__(self, evaluate_metrics='auto', evaluate_prediction_dir=None):
        self.evaluate_prediction_dir = evaluate_prediction_dir

    def experiment_end(self, exp, elapsed):
        from hypernets.utils import logging
        logger = logging.get_logger(__name__)

        if exp.y_eval is None or exp.X_eval is None:  # skip it there is eval_data
            logger.warning("y_eval or X_eval is None, skip to evaluate")
            return

        # predict X
        t_predict_start = time.time()
        y_eval_pred = exp.model_.predict(exp.X_eval)
        predict_elapsed = time.time() - t_predict_start
        if hasattr(exp.model_, 'predict_proba'):
            t_predict_proba_start = time.time()
            try:
                y_eval_proba = exp.model_.predict_proba(exp.X_eval)
            except Exception as e:
                logger.exception(e)
                y_eval_proba = None
            finally:
                predict_proba_elapsed = time.time() - t_predict_proba_start
        else:
            predict_proba_elapsed = None
            y_eval_proba = None

        def to_pkl(obj, path):
            path = os.path.abspath(path)
            assert path
            if os.path.exists(path):
                logger.warning(f"persist path is already exists and will be overwritten {path} ")
            with open(path, 'wb') as f:
                joblib.dump(obj, f)
            logger.info(f'persist predictions object to path {path}')

        # persist predictions
        write_dir = self.evaluate_prediction_dir
        if write_dir is not None:
            if not os.path.exists(write_dir):
                logger.info(f"create prediction persist directory: {write_dir}")
                os.makedirs(write_dir, exist_ok=True)

            persist_pred_path = os.path.join(write_dir, 'predict.pkl')
            to_pkl(y_eval_pred, persist_pred_path)
            if y_eval_proba is not None:
                persist_proba_path = os.path.join(write_dir, 'predict_proba.pkl')
                to_pkl(y_eval_proba, persist_proba_path)

        #  evaluate model
        DIGITS = 4
        y_test = df_utils.as_array(exp.y_eval)
        y_pred = df_utils.as_array(y_eval_pred)

        classification_report_data = None
        evaluation_metrics_data = None
        confusion_matrix_data = None

        if exp.task in [const.TASK_MULTICLASS, const.TASK_BINARY]:
            classification_report_data = classification_report(y_test, y_pred, output_dict=True, digits=DIGITS)

            labels = unique_labels(y_test, y_pred)
            confusion_matrix_data = ConfusionMatrixMeta(confusion_matrix(y_test, y_pred, labels=labels), labels)
        elif exp.task in [const.TASK_REGRESSION]:
            from hypernets.tabular.metrics import calc_score
            evaluation_metrics_data = calc_score(y_test, y_pred, y_proba=None, metrics=('mse', 'mae', 'rmse', 'r2'),
                                                 task=const.TASK_REGRESSION, pos_label=None, classes=None, average=None)
            evaluation_metrics_data['explained_variance'] = \
                sk_metrics.explained_variance_score(y_true=y_test, y_pred=y_pred)

        exp.evaluation_ = {
            'prediction_elapsed': (predict_elapsed, predict_proba_elapsed),
            'evaluation_metrics': evaluation_metrics_data,
            'confusion_matrix': confusion_matrix_data,
            'classification_report': classification_report_data
        }


class ResourceUsageMonitor(Thread):
    STATUS_READY = 0
    STATUS_RUNNING = 1
    STATUS_STOP = 2

    def __init__(self, interval=30):
        # auto exit if all work thread finished
        super(ResourceUsageMonitor, self).__init__(name=self.__class__.__name__, daemon=True)
        self.interval = interval

        self._timer_status_lock = RLock()
        self._timer_status = 0  # 0->ready, 1->running, 2->stop
        self._process = psutil.Process(os.getpid())
        self._data = []

    def _change_status(self, status):
        self._timer_status_lock.acquire(blocking=True, timeout=30)
        try:
            self._timer_status = status  # state machine: check status
        except Exception as e:
            pass
        finally:
            self._timer_status_lock.release()

    def start_watch(self):
        if self._timer_status != ResourceUsageMonitor.STATUS_READY:
            raise Exception(f"Illegal status: {self._timer_status}, "
                            f"only at '{ResourceUsageMonitor.STATUS_READY}' is available")
        self.start()

    def stop_watch(self):
        self._change_status(ResourceUsageMonitor.STATUS_STOP)

    def run(self) -> None:
        while self._timer_status != ResourceUsageMonitor.STATUS_STOP:
            cpu_percent = self._process.cpu_percent()
            mem_percent = self._process.memory_percent()
            self._data.append((datetime.now(), cpu_percent, mem_percent))
            time.sleep(self.interval)

    @property
    def data(self):
        return self._data  # [(datetime, cpu, mem)]


class MLReportCallback(ExperimentCallback):

    def __init__(self, render: ReportRender, sample_interval=30):
        self.render = render
        self._rum = ResourceUsageMonitor(interval=sample_interval)

        # self.render_options = render_options if not None else {}
        self.experiment_meta_ = None

    def experiment_start(self, exp):
        self._rum.start_watch()

    def experiment_end(self, exp, elapsed):
        # get experiment meta and render to excel
        self.experiment_meta_ = ExperimentExtractor(exp, self._rum.data).extract()
        self.render.render(self.experiment_meta_)

    def experiment_break(self, exp, error):
        pass  # TODO

    def step_start(self, exp, step):
        pass

    def step_progress(self, exp, step, progress, elapsed, eta=None):
        pass

    def step_end(self, exp, step, output, elapsed):
        pass

    def step_break(self, exp, step, error):
        pass

    def __getstate__(self):
        states = dict(self.__dict__)
        if '_rum' in states:
            del states['_rum']

        return states


class ActionType:
    ExperimentStart = 'experimentStart'
    ExperimentBreak = 'experimentBreak'
    ExperimentEnd = 'experimentEnd'
    StepStart = 'stepStart'
    StepBreak = 'stepBreak'
    StepEnd = 'stepEnd'
    EarlyStopped = 'earlyStopped'
    TrialEnd = 'trialEnd'


class ABSExpVisHyperModelCallback(Callback, metaclass=abc.ABCMeta):

    def __init__(self):
        super(Callback, self).__init__()
        self.max_trials = None
        self.current_running_step_index = None
        self.exp_id = None

    def set_exp_id(self, exp_id):
        self.exp_id = exp_id

    def set_current_running_step_index(self, value):
        self.current_running_step_index = value

    def assert_ready(self):
        assert self.exp_id is not None
        assert self.current_running_step_index is not None

    def on_search_start(self, hyper_model, X, y, X_eval, y_eval, cv,
                        num_folds, max_trials, dataset_id, trial_store, **fit_kwargs):
        self.max_trials = max_trials  # to record trail summary info

    @staticmethod
    def get_early_stopping_status_data(hyper_model):
        """ Return early stopping data if triggered """
        # check whether end cause by early stopping
        for c in hyper_model.callbacks:
            if isinstance(c, EarlyStoppingCallback):
                if c.triggered:
                    if c.start_time is not None:
                        elapsed_time = time.time() - c.start_time
                    else:
                        elapsed_time = None
                    ess = EarlyStoppingStatusMeta(c.best_reward, c.best_trial_no, c.counter_no_improvement_trials,
                                                  c.triggered, c.triggered_reason, elapsed_time)
                    return ess
        return None

    def on_search_end(self, hyper_model):
        self.assert_ready()
        early_stopping_data = self.get_early_stopping_status_data(hyper_model)
        self.on_search_end_(hyper_model, early_stopping_data)

    def on_search_end_(self, hyper_model, early_stopping_data):
        pass

    @staticmethod
    def get_space_params(space):
        params_dict = {}
        for hyper_param in space.get_assigned_params():
            param_name = hyper_param.alias
            param_value = hyper_param.value
            if param_name is not None and param_value is not None:
                params_dict[param_name] = str(param_value)
        return params_dict


class ABSExpVisExperimentCallback(ExperimentCallback, metaclass=abc.ABCMeta):

    def __init__(self, hyper_model_callback_cls: Type[ABSExpVisHyperModelCallback], **kwargs):
        self.hyper_model_callback_cls = hyper_model_callback_cls

    @staticmethod
    def get_step(experiment, step_name):
        for i, step in enumerate(experiment.steps):
            if step.name == step_name:
                return i, step
        return -1, None

    @classmethod
    def get_step_index(cls, experiment, step_name):
        return cls.get_step(experiment, step_name)[0]

    def experiment_start(self, exp):
        self.setup_hyper_model_callback(exp, -1)
        d = ExperimentExtractor(exp).extract()
        self.experiment_start_(exp, d)

    @abc.abstractmethod
    def experiment_start_(self, exp, experiment_data):
        raise NotImplemented

    def _find_hyper_model_callback(self, exp):
        for callback in exp.hyper_model.callbacks:
            if isinstance(callback, self.hyper_model_callback_cls):
                return callback
        return exp

    def setup_hyper_model_callback(self, exp, step_index):
        hyper_model_callback = self._find_hyper_model_callback(exp)
        assert hyper_model_callback
        hyper_model_callback.set_current_running_step_index(step_index)
        hyper_model_callback.set_exp_id(id(exp))

    def step_start(self, exp, step):
        step_name = step
        step_index = self.get_step_index(exp, step_name)
        self.setup_hyper_model_callback(exp, step_index)
        payload = {
            'index': step_index,
            'status': StepMeta.STATUS_PROCESS,
            'start_datetime': time.time(),
        }
        self.step_start_(exp, step, payload)

    @abc.abstractmethod
    def step_start_(self, exp, step, d):
        raise NotImplemented

    def step_end(self, exp, step, output, elapsed):
        step_name = step
        step = exp.get_step(step_name)

        step.done_time = time.time()  # fix done_time is none
        step_index = self.get_step_index(exp, step_name)
        step_meta_dict = ExperimentExtractor.extract_step(step_index, step).to_dict()

        self.step_end_(exp, step, output, elapsed, step_meta_dict)

    @abc.abstractmethod
    def step_end_(self, exp, step, output, elapsed, step_meta_dict):
        raise NotImplemented

    def step_break(self, exp, step, error):
        step_name = step
        step_index = self.get_step_index(exp, step_name)
        payload = {
            'index': step_index,
            'extension': {
                'reason': str(error)
            },
            'status': StepMeta.STATUS_ERROR
        }
        self.step_break_(exp, step, error, payload)

    @abc.abstractmethod
    def step_break_(self, exp, step, error, payload):
        raise NotImplemented
