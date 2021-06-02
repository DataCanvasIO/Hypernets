# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import time

from hypernets.dispatchers.cfg import DispatchCfg
from hypernets.utils import logging

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

    def show_details(self):
        Missing_y = self.y_train.isnull().sum()
        Unique_y = len(self.y_train.unique())
        Freq_y = self.y_train.value_counts()[0]
        
        if self.y_train.dtypes != 'int64' and self.y_train.dtypes != 'float64':
            max_y = None
        else:
            max_y = max(self.y_train)
        if self.y_train.dtypes != 'int64' and self.y_train.dtypes != 'float64':
            min_y = None
        else:
            min_y = min(self.y_train)
        if self.y_train.dtypes != 'int64' and self.y_train.dtypes != 'float64':
            mean_y = None
        else:
            mean_y = pd.Series.mean(self.y_train)
        if self.y_train.dtypes != 'int64' and self.y_train.dtypes != 'float64':
            Stdev_y = None
        else:
            Stdev_y = self.y_train.std()
        
        dtype2usagetype = {'object':'str', 'int64':'int', 'float64':'float', 'datetime64[ns]':'date', 'timedelta64[ns]':'date'}
        datatype_y = dtype2usagetype[str(self.y_train.dtypes)]

        if datatype_y == 'str' or datatype_y == 'date':
            disc_y_num = 10
            Freq_y = y_train.value_counts()[0:disc_y_num-1]
            names_y = Freq_y.keys()

        if datatype_y == 'int' or datatype_y == 'float':
            cont_y_num = 10
            interval = max(y_train)/10
            intervals = [0]
            for i in range(1,cont_y_num+1):
                intervals.append(i*interval)
            Freq_y = pd.cut(y_train, intervals).value_counts()[0:cont_y_num-1]
            names_y = Freq_y.keys()
        
        details_dict = {
            'yCharacters':{
                'tasktypeY':self.task,
                'maxY':max_y,
                'minY':min_y, 
                'meanY':mean_y, 
                'StdevY':Stdev_y, 
                'datatypeY':datatype_y,
                'MissingY':Missing_y,
                'UniqueY':Unique_y,
                'FreqY':Freq_y,
                'datatypeY':datatype_y
            },
            'shapes':{
                'shapeXTrain':self.X_train.shape, 
                'shapeYTrain':self.y_train.shape, 
                'shapeXEval':self.X_eval.shape, 
                'shapeYEval':self.y_eval.shape, 
                'shapeXTest':self.X_test.shape
            },
            'yDistribution':{
                'discYNum':disc_y_num,
                'contYNum':cont_y_num,
                'FreqY':Freq_y,
                'namesY':names_y
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
