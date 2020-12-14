# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from . import ExperimentCallback


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
