# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import time

from hypernets.dispatchers.cfg import DispatchCfg
from hypernets.utils import logging

import numpy as np
import pandas as pd
from dask import dataframe as dd

logger = logging.get_logger(__name__)


class ExperimentCallback():
    def experiment_start(self, exp):
        pass

    def experiment_end(self, exp, elapsed):
        pass

    def experiment_break(self, exp, error):
        pass

    def step_start(self, exp, step):
        pass

    def step_progress(self, exp, step, progress, elapsed, eta=None):
        pass

    def step_end(self, exp, step, output, elapsed):
        pass

    def step_break(self, exp, step, error):
        pass


class Experiment(object):
    def __init__(self, hyper_model, X_train, y_train, X_eval=None, y_eval=None, X_test=None, eval_size=0.3,
                 task=None, id=None, callbacks=None, random_state=9527):
        self.task = task
        self.id = id
        self.title = None
        self.description = None
        self.dataset_id = None
        self.path = None
        self.current_step = None
        self.step_start_time = None

        self.hyper_model = hyper_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.X_test = X_test

        if X_eval is not None and X_test is None:
            self.X_test = X_eval

        self.eval_size = eval_size
        self.callbacks = callbacks if callbacks is not None else []
        self.random_state = random_state

        self.start_time = None
        self.end_time = None

    def get_data_character(self):

        dtype2usagetype = {'object':'str', 'int64':'int', 'float64':'float', 'datetime64[ns]':'date', 'timedelta64[ns]':'date'}
        
        self.task, _ = self.hyper_model.infer_task_type(self.y_train)

        if isinstance(self.y_train, pd.core.series.Series):
            datatype_y = dtype2usagetype[str(self.y_train.dtypes)]

            Missing_y = self.y_train.isnull().sum().tolist()
            Unique_y = len(self.y_train.unique())
            Freq_y = self.y_train.value_counts()[0].tolist()
            
            if self.task == 'regression':
                max_y = max(self.y_train)
                min_y = min(self.y_train)
                mean_y = pd.Series.mean(self.y_train)
                Stdev_y = self.y_train.std()
            else:
                max_y = None
                min_y = None
                mean_y = None
                Stdev_y = None

            # if self.y_train.dtypes != 'int64' and self.y_train.dtypes != 'float64':
            #     max_y = None
            #     min_y = None
            #     mean_y = None
            #     Stdev_y = None
            # else:
            #     max_y = max(self.y_train)
            #     min_y = min(self.y_train)
            #     mean_y = pd.Series.mean(self.y_train)
            #     Stdev_y = self.y_train.std()

            if self.task == 'regression':
                Freq_y = None
                disc_y_num = None
                cont_y_num = 10
                interval = (max_y-min_y)/cont_y_num
                intervals = np.linspace(min_y, max_y, cont_y_num+1)
                Freqs_y = pd.cut(self.y_train, intervals).value_counts()[0:cont_y_num]
                count = list(Freqs_y)
                region = list(map(lambda x: [x.left, x.right], list(Freqs_y.keys())))
                target_distribution = {'count':count, 'region':region}
            else:
                cont_y_num = None
                disc_y_num = 10
                target_distribution = dict(self.y_train.value_counts()[0:disc_y_num])
                for key in target_distribution:
                    target_distribution[key] = int(target_distribution[key])

            # if datatype_y == 'str' or datatype_y == 'date':
            #     cont_y_num = None
            #     disc_y_num = 10
            #     target_distribution = dict(self.y_train.value_counts()[0:disc_y_num])
            #     for key in target_distribution:
            #         target_distribution[key] = int(target_distribution[key])

            # if datatype_y == 'int' or datatype_y == 'float':
            #     Freq_y = None
            #     disc_y_num = None
            #     cont_y_num = 10
            #     interval = (max(self.y_train)-min(self.y_train))/cont_y_num
            #     intervals = np.linspace(min(self.y_train), max(self.y_train), cont_y_num+1)
            #     Freqs_y = pd.cut(self.y_train, intervals).value_counts()[0:cont_y_num]
            #     count = list(Freqs_y)
            #     region = list(map(lambda x: [x.left, x.right], list(Freqs_y.keys())))
            #     target_distribution = {'count':count, 'region':region}
            
            shape_x_train = list(self.X_train.shape)
            shape_y_train = list(self.y_train.shape)
            if self.X_eval == None:
                shape_x_eval = []
            else:
                shape_x_eval = list(self.X_eval.shape)

            if self.y_eval == None:
                shape_y_eval = []
            else:
                shape_y_eval = list(self.y_eval.shape)

            if self.X_test == None:
                shape_x_test = []
            else:
                shape_x_test = list(self.X_test.shape)

            # # 检查类型
            # print(type(int(Freq_y)))
            # print(type(Unique_y))
            # print(type(Missing_y))
            # keyyy = list(target_distribution.keys())[0]
            # print(type(target_distribution[keyyy]))
            # print(type(list(self.X_train.shape)[0]))

        else:
            datatype_y = dtype2usagetype[str(self.y_train.dtype)]

            Missing_y = self.y_train.isnull().compute().tolist().count(True)
            Unique_y = len(self.y_train.unique().compute().tolist())
            Freq_y = self.y_train.value_counts().compute().tolist()[0]

            if self.y_train.dtype != 'int64' and self.y_train.dtype != 'float64':
                max_y = None
                min_y = None
                mean_y = None
                Stdev_y = None
            else:
                max_y = self.y_train.max().compute().tolist()
                min_y = self.y_train.min().compute().tolist()
                mean_y = self.y_train.mean().compute().tolist()
                Stdev_y = self.y_train.std().compute().tolist()
            
            if datatype_y == 'str' or datatype_y == 'date':
                cont_y_num = None
                disc_y_num = 10
                target_distribution = dict(dd.compute(self.y_train.value_counts())[0])
                for key in target_distribution:
                    target_distribution[key] = int(target_distribution[key])

            if datatype_y == 'int' or datatype_y == 'float':
                Freq_y = None
                disc_y_num = None
                cont_y_num = 10
                interval = (max_y-min_y)/cont_y_num
                intervals = np.linspace(min_y, max_y, cont_y_num+1)
                Freqs_y = pd.cut(self.y_train, intervals).value_counts()[0:cont_y_num]
                count = list(Freqs_y)
                region = list(map(lambda x: [x.left, x.right], list(Freqs_y.keys())))
                target_distribution = {'count':count, 'region':region}

            shape_x_train = list(self.X_train.shape)
            for idx, num in enumerate(shape_x_train):
                if isinstance(num, int):
                    continue
                else:
                    shape_x_train[idx] = num.compute()

            shape_y_train = list(map(lambda x: x.compute(), list(self.y_train.shape)))

            if self.X_eval == None:
                shape_x_eval = []
            else:
                shape_x_eval = list(self.X_eval.shape)
                for idx, num in enumerate(shape_x_eval):
                    if isinstance(num, int):
                        continue
                    else:
                        shape_x_eval[idx] = num.compute()
            
            if self.y_eval == None:
                shape_y_eval = []
            else:
                shape_y_eval = list(self.y_eval.shape)
                for idx, num in enumerate(shape_y_eval):
                    if isinstance(num, int):
                        continue
                    else:
                        shape_y_eval[idx] = num.compute()

            if self.X_test == None:
                shape_x_test = []
            else:
                shape_x_test = list(self.X_test.shape)
                for idx, num in enumerate(shape_x_test):
                    if isinstance(num, int):
                        continue
                    else:
                        shape_x_test[idx] = num.compute()
        
        details_dict = {
            'experimentType': 'base',
            'target':{
                'name':'y',
                'taskType':self.task,
                'freq':Freq_y,
                'unique':Unique_y,
                'missing':Missing_y,
                'mean':mean_y, 
                'min':min_y,
                'max':max_y,
                'stdev':Stdev_y, 
                'dataType':datatype_y
            },
            'targetDistribution': target_distribution,
            'datasetShape':{
                'X_train':shape_x_train, 
                'y_train':shape_y_train, 
                'X_eval':shape_x_eval, 
                'y_eval':shape_y_eval, 
                'X_test':shape_x_test
            }
            }

        return details_dict

    def run(self, **kwargs):
        self.start_time = time.time()

        DispatchCfg.experiment = str(self.id) if self.id is not None else ''

        try:
            if self.task is None:
                self.task, _ = self.hyper_model.infer_task_type(self.y_train)

            for callback in self.callbacks:
                callback.experiment_start(self)

            model = self.train(self.hyper_model, self.X_train, self.y_train, self.X_test, X_eval=self.X_eval,
                               y_eval=self.y_eval, **kwargs)

            self.end_time = time.time()

            for callback in self.callbacks:
                callback.experiment_end(self, self.elapsed)
            return model
        except Exception as e:
            import traceback
            msg = f'ExperiementID:[{self.id}] - {self.current_step}: {e}\n{traceback.format_exc()}'
            logger.error(msg)
            for callback in self.callbacks:
                callback.experiment_break(self, msg)

    def step_start(self, step):
        if self.current_step is not None:
            raise RuntimeError(
                f'Make sure `step_end()` is called at the end of the previous step. `step`:{step}, `current_step`:{self.current_step}')

        self.current_step = step
        self.step_start_time = time.time()
        for callback in self.callbacks:
            callback.step_start(self, step)

    def step_end(self, output=None):
        if self.current_step is None:
            raise RuntimeError('Make sure `step_start()` is called at the start of the current step.')
        elapsed = time.time() - self.step_start_time
        for callback in self.callbacks:
            callback.step_end(self, self.current_step, output, elapsed)
        self.current_step = None
        self.step_start_time = None

    def step_break(self, error=None):
        if self.current_step is None:
            raise RuntimeError('Make sure `step_start()` is called at the start of the current step.')

        for callback in self.callbacks:
            callback.step_break(self, self.current_step, error)
        self.current_step = None
        self.step_start_time = None

    def step_progress(self, progress, eta=None):
        elapsed = time.time() - self.step_start_time
        for callback in self.callbacks:
            callback.step_progress(self, self.current_step, progress, elapsed, eta)

    def train(self, hyper_model, X_train, y_train, X_test, X_eval=None, y_eval=None, **kwargs):
        """Run an experiment

        Arguments
        ---------
        hyper_model : HyperModel
        X_train :
        y_train :
        X_test :
        X_eval :
        y_eval :
        eval_size :
        """
        raise NotImplementedError

    def export_model(self):
        raise NotImplementedError

    @property
    def elapsed(self):
        if self.start_time is None or self.end_time is None:
            return -1
        else:
            return self.end_time - self.start_time
