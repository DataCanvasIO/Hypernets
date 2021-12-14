# -*- coding:utf-8 -*-
__author__ = 'yangjian'

from ._extractor import ConfusionMatrixMeta

"""

"""
import time
import os
from datetime import datetime
from collections import OrderedDict
from threading import Thread, RLock
import math

import pandas as pd
import joblib
import psutil
from sklearn import metrics as sk_metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from IPython.display import display, display_markdown

from hypernets.tabular import get_tool_box
from hypernets.utils import const
from hypernets.experiment import ExperimentExtractor
from . import ExperimentCallback
from .compete import StepNames
from .report import ReportRender


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

            import seaborn as sns
            import matplotlib.pyplot as plt
            # Draw Plot
            plt.figure(figsize=(8, 4), dpi=80)
            sns.kdeplot(proba, shade=True, color="g", label="Proba", alpha=.7, bw_adjust=0.01)
            # Decoration
            plt.title('Density Plot of Probability', fontsize=22)
            plt.legend()
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
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import LabelEncoder
            if exp.task == const.TASK_REGRESSION:
                # Draw Plot
                plt.figure(figsize=(8, 4), dpi=80)
                sns.kdeplot(y_train.dropna(), shade=True, color="g", label="Proba", alpha=.7, bw_adjust=0.01)
            else:
                le = LabelEncoder()
                y = le.fit_transform(y_train.dropna())
                # Draw Plot
                plt.figure(figsize=(8, 4), dpi=80)
                sns.distplot(y, kde=False, color="g", label="y")
            # Decoration
            plt.title('Distribution of y', fontsize=22)
            plt.legend()
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

        self._eval_df_shape = None
        self._eval_elapse = None

    def experiment_start(self, exp):
        """
            exp.evaluation:
                {
                    'metrics': {
                        'f1': 0.1,
                    },
                    "classification_report": {
                        "0": {
                            'f1': 0.1,
                        }
                    },
                    "confusion_matrix": [[1,2], [3,4]],
                    "timing": {
                        'predict': 100,
                        'predict_proba': 100
                    }
                }
        Parameters
        ----------
        exp

        Returns
        -------

        """
        # define attrs
        exp.y_eval_proba = None
        exp.y_eval_pred = None
        exp.evaluation = {}

    @staticmethod
    def to_prediction(y_score):
        pass

    @staticmethod
    def _persist(obj, obj_name, path):
        if path is not None:
            if os.path.exists(path):
                print(f"[WARNING] persist path is already exists: {path} ")
            print(f"Persist '{obj_name}' result to '{path}' ")
            with open(path, 'wb') as f:
                joblib.dump(obj, f)

    def experiment_end(self, exp, elapsed):
        # Attach prediction
        if exp.y_eval is not None and exp.X_eval is not None:
            _t = time.time()
            exp.y_eval_pred = exp.model_.predict(exp.X_eval)  # TODO: try to use proba
            if hasattr(exp.model_, 'predict_proba'):
                try:
                    exp.y_eval_proba = exp.model_.predict_proba(exp.X_eval)
                except Exception as e:
                    print("predict_proba failed", e)
            exp.evaluation['timing'] = {'predict': time.time() - _t, 'predict_proba': 0}
        if self.evaluate_prediction_dir is not None:
            persist_pred_path = os.path.join(self.evaluate_prediction_dir, 'predict.pkl')
            persist_proba_path = os.path.join(self.evaluate_prediction_dir, 'predict_proba.pkl')
            self._persist(exp.y_eval_proba, 'y_eval_proba', persist_pred_path)
            self._persist(exp.y_eval_pred, 'y_eval_pred', persist_proba_path)
        # TODO: add evaluation


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
        self._experiment_meta = None

    def experiment_start(self, exp):
        self._rum.start_watch()
        # define attrs
        exp.y_eval_proba = None
        exp.y_eval_pred = None
        exp.evaluation = {}

    def experiment_end(self, exp, elapsed):
        # 1. Mock full experiment: Add evaluation
        confusion_matrix_result = None
        DIGITS = 4
        if exp.y_eval_pred is not None:
            y_test = exp.y_eval
            y_pred = exp.y_eval_pred
            if exp.task in [const.TASK_MULTICLASS, const.TASK_BINARY]:
                evaluation_result = classification_report(y_test, y_pred, output_dict=True, digits=DIGITS)
                labels = unique_labels(y_test, y_pred)
                confusion_matrix_data = confusion_matrix(y_test, y_pred, labels=labels)
                confusion_matrix_result = ConfusionMatrixMeta(confusion_matrix_data, labels)
            elif exp.task in [const.TASK_REGRESSION]:
                explained_variance = round(sk_metrics.explained_variance_score(y_true=y_test, y_pred=y_pred), DIGITS)
                neg_mean_absolute_error = round(sk_metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred), DIGITS)
                neg_mean_squared_error = round(sk_metrics.mean_squared_error(y_true=y_test, y_pred=y_pred), DIGITS)
                rmse = round(math.sqrt(neg_mean_squared_error), DIGITS)
                neg_median_absolute_error = round(sk_metrics.median_absolute_error(y_true=y_test, y_pred=y_pred),
                                                  DIGITS)
                r2 = round(sk_metrics.r2_score(y_true=y_test, y_pred=y_pred), DIGITS)
                if (y_test >= 0).all() and (y_pred >= 0).all():
                    neg_mean_squared_log_error = round(sk_metrics.mean_squared_log_error(y_true=y_test, y_pred=y_pred),
                                                       DIGITS)
                else:
                    neg_mean_squared_log_error = None
                evaluation_result = {
                    "explained_variance": explained_variance,
                    "neg_mean_absolute_error": neg_mean_absolute_error,
                    "neg_mean_squared_error": neg_mean_squared_error,
                    "rmse": rmse,
                    "neg_mean_squared_log_error": neg_mean_squared_log_error,
                    "r2": r2,
                    "neg_median_absolute_error": neg_median_absolute_error
                }
            else:
                evaluation_result = None
        else:
            evaluation_result = {}

        # 2. get experiment meta and render to excel
        self._experiment_meta = ExperimentExtractor(exp, evaluation_result,
                                                    confusion_matrix_result, self._rum.data).extract()
        self.render.render(self._experiment_meta)

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

