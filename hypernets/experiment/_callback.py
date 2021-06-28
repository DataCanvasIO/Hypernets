# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from collections import OrderedDict

import pandas as pd
from IPython.display import display, display_markdown

from hypernets.tabular import dask_ex as dex
from hypernets.utils import const
from . import ExperimentCallback
from .compete import StepNames


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

        if dex.exist_dask_object(X_train, y_train, X_test, X_eval, y_eval):
            display_data = (dex.compute(X_train.shape)[0],
                            dex.compute(y_train.shape)[0],
                            dex.compute(X_eval.shape)[0] if X_eval is not None else None,
                            dex.compute(y_eval.shape)[0] if y_eval is not None else None,
                            dex.compute(X_test.shape)[0] if X_test is not None else None,
                            exp.task if exp.task == const.TASK_REGRESSION
                            else f'{exp.task}({dex.compute(y_train.nunique())[0]})')
        else:
            display_data = (X_train.shape,
                            y_train.shape,
                            X_eval.shape if X_eval is not None else None,
                            y_eval.shape if y_eval is not None else None,
                            X_test.shape if X_test is not None else None,
                            exp.task if exp.task == const.TASK_REGRESSION
                            else f'{exp.task}({y_train.nunique()})')
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

            le = LabelEncoder()
            y = le.fit_transform(y_train.dropna())
            # Draw Plot
            plt.figure(figsize=(8, 4), dpi=80)
            sns.distplot(y, kde=False, color="g", label="y")
            # Decoration
            plt.title('Distribution of y', fontsize=12)
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
