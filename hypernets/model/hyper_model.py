# -*- coding:utf-8 -*-
"""

"""
import time
import traceback
from collections import UserDict

from ..core.meta_learner import MetaLearner
from ..core.trial import Trial, TrialHistory, DiskTrialStore
from ..discriminators import UnPromisingTrial
from ..dispatchers import get_dispatcher
from ..tabular import get_tool_box
from ..utils import logging, const, to_repr

logger = logging.get_logger(__name__)


class HyperModel:
    def __init__(self, searcher, dispatcher=None, callbacks=None, reward_metric=None, task=None, discriminator=None):
        """

        :param searcher:
        :param dispatcher:
        :param callbacks:
        :param reward_metric:
        :param task:
        """
        self.searcher = searcher
        self.dispatcher = dispatcher
        self.callbacks = callbacks if callbacks is not None else []
        self.reward_metric = reward_metric
        self.history = TrialHistory(searcher.optimize_direction)
        self.task = task
        self.discriminator = discriminator
        if self.discriminator:
            self.discriminator.bind_history(self.history)

    def _get_estimator(self, space_sample):
        raise NotImplementedError

    def load_estimator(self, model_file):
        raise NotImplementedError

    def _run_trial(self, space_sample, trial_no, X, y, X_eval, y_eval, cv=False, num_folds=3, model_file=None,
                   **fit_kwargs):

        start_time = time.time()
        estimator = self._get_estimator(space_sample)
        if self.discriminator:
            estimator.set_discriminator(self.discriminator)

        for callback in self.callbacks:
            try:
                callback.on_build_estimator(self, space_sample, estimator, trial_no)
            except Exception as e:
                logger.warn(e)

        metrics = fit_kwargs.pop('metrics') if 'metrics' in fit_kwargs else None
        if metrics is not None:
            assert isinstance(metrics, (tuple, list)), 'metrics should be list or tuple'
            if not any(map(lambda _: _ is self.reward_metric, metrics)):
                metrics = list(metrics) + [self.reward_metric]
        else:
            metrics = [self.reward_metric]

        succeeded = False
        scores = None
        oof = None
        oof_scores = None
        try:
            if cv:
                scores, oof, oof_scores = estimator.fit_cross_validation(X, y, stratified=True, num_folds=num_folds,
                                                                         shuffle=False, random_state=9527,
                                                                         metrics=metrics,
                                                                         **fit_kwargs)
            else:
                estimator.fit(X, y, **fit_kwargs)
            succeeded = True
        except UnPromisingTrial as e:
            logger.info(f'{e}')
        except Exception as e:
            logger.error(f'run_trail failed! trail_no={trial_no}')
            track = traceback.format_exc()
            logger.error(track)

        if succeeded:
            if scores is None:
                scores = estimator.evaluate(X_eval, y_eval, metrics=metrics, **fit_kwargs)
            reward = self._get_reward(scores, self.reward_metric)

            if model_file is None or len(model_file) == 0:
                model_file = '%05d_%s.pkl' % (trial_no, space_sample.space_id)
            estimator.save(model_file)

            elapsed = time.time() - start_time

            trial = Trial(space_sample, trial_no, reward, elapsed, model_file, succeeded)
            trial.iteration_scores = estimator.get_iteration_scores()
            trial.memo['scores'] = scores
            if oof is not None and self._is_memory_enough(oof):
                trial.memo['oof'] = oof
            if oof_scores is not None:
                trial.memo['oof_scores'] = oof_scores

            # improved = self.history.append(trial)

            self.searcher.update_result(space_sample, reward)
        else:
            elapsed = time.time() - start_time
            trial = Trial(space_sample, trial_no, 0, elapsed, succeeded=succeeded)

            if self.history is not None:
                t = self.history.get_worst()
                if t is not None:
                    self.searcher.update_result(space_sample, t.reward)

        return trial

    @staticmethod
    def _is_memory_enough(oof):
        tb = get_tool_box(oof)
        free = tb.memory_free() / tb.memory_total()
        return free > 0.618

    def _get_reward(self, value, key=None):
        def cast_float(value):
            try:
                fv = float(value)
                return fv
            except TypeError:
                return None

        if key is None:
            key = 'reward'
        elif callable(key) and hasattr(key, '__name__'):
            key = key.__name__

        fv = cast_float(value)
        if fv is not None:
            reward = fv
        elif isinstance(value, (dict, UserDict)) and key in value and cast_float(value[key]) is not None:
            reward = cast_float(value[key])
        else:
            raise ValueError(
                f'[value] should be a numeric or a dict which has a key named "{key}" whose value is a numeric.')
        return reward

    def get_best_trial(self):
        return self.history.get_best()

    @property
    def best_reward(self):
        best = self.get_best_trial()
        if best is not None:
            return best.reward
        else:
            return None

    @property
    def best_trial_no(self):
        best = self.get_best_trial()
        if best is not None:
            return best.trial_no
        else:
            return None

    def get_top_trials(self, top_n):
        return self.history.get_top(top_n)

    def _before_search(self):
        pass

    def _after_search(self, last_trial_no):
        pass

    def search(self, X, y, X_eval, y_eval, cv=False, num_folds=3, max_trials=10, dataset_id=None, trial_store=None,
               **fit_kwargs):
        """
        :param X: Pandas or Dask DataFrame, feature data for training
        :param y: Pandas or Dask Series, target values for training
        :param X_eval: (Pandas or Dask DataFrame) or None, feature data for evaluation
        :param y_eval: (Pandas or Dask Series) or None, target values for evaluation
        :param cv: Optional, int(default=False), If set to `true`, use cross-validation instead of evaluation set reward to guide the search process
        :param num_folds: Optional, int(default=3), Number of cross-validated folds, only valid when cv is true
        :param max_trials: Optional, int(default=10), The upper limit of the number of search trials, the search process stops when the number is exceeded
        :param dataset_id:
        :param trial_store:
        :param fit_kwargs: Optional, dict, parameters for fit method of model
        :return:
        """
        if self.task is None or self.task == const.TASK_AUTO:
            self.task, _ = self.infer_task_type(y)
        if self.task not in [const.TASK_BINARY, const.TASK_MULTICLASS, const.TASK_REGRESSION, const.TASK_MULTILABEL]:
            logger.warning(f'Unexpected task "{self.task}"')

        if dataset_id is None:
            dataset_id = self.generate_dataset_id(X, y)
        if isinstance(trial_store, str):
            trial_store = DiskTrialStore(trial_store)
        if self.searcher.use_meta_learner:
            self.searcher.set_meta_learner(MetaLearner(self.history, dataset_id, trial_store))

        self._before_search()

        dispatcher = self.dispatcher if self.dispatcher else get_dispatcher(self)

        for callback in self.callbacks:
            try:
                callback.on_search_start(self, X, y, X_eval, y_eval,
                                         cv, num_folds, max_trials, dataset_id, trial_store,
                                         **fit_kwargs)
            except Exception as e:
                logger.warn(e)

        try:
            trial_no = dispatcher.dispatch(self, X, y, X_eval, y_eval,
                                           cv, num_folds, max_trials, dataset_id, trial_store,
                                           **fit_kwargs)

            for callback in self.callbacks:
                try:
                    callback.on_search_end(self)
                except Exception as e:
                    logger.warn(e)
        except Exception as e:
            cb_ex = False
            for callback in self.callbacks:
                try:
                    callback.on_search_error(self)
                except Exception as ce:
                    logger.warn(ce)
                    cb_ex = True
            if cb_ex:
                raise e
            else:
                raise

        self._after_search(trial_no)

    def generate_dataset_id(self, X, y):
        if hasattr(X, 'shape') and len(getattr(X, 'shape')) == 2:
            tb = get_tool_box(X, y)
            sign = tb.data_hasher()([X, y])
        else:
            import hashlib
            repr = ''
            if X is not None:
                if isinstance(X, list):
                    repr += f'X len({len(X)})|'
                if hasattr(X, 'shape'):
                    repr += f'X shape{X.shape}|'
                if hasattr(X, 'dtypes'):
                    repr += f'x.dtypes({list(X.dtypes)})|'

            if y is not None:
                if isinstance(y, list):
                    repr += f'y len({len(y)})|'
                if hasattr(y, 'shape'):
                    repr += f'y shape{y.shape}|'

                if hasattr(y, 'dtype'):
                    repr += f'y.dtype({y.dtype})|'

            sign = hashlib.md5(repr.encode('utf-8')).hexdigest()
        return sign

    def final_train(self, space_sample, X, y, **kwargs):
        estimator = self._get_estimator(space_sample)
        estimator.set_discriminator(None)
        estimator.fit(X, y, **kwargs)
        return estimator

    def export_configuration(self, trials):
        configurations = []
        for trial in trials:
            configurations.append(self.export_trial_configuration(trial))
        return configurations

    def export_trial_configuration(self, trial):
        raise NotImplementedError

    def infer_task_type(self, y):
        return get_tool_box(y).infer_task_type(y)

    def plot_hyperparams(self, destination='notebook', output='hyperparams.html'):
        return self.history.plot_hyperparams(destination, output)

    def __repr__(self):
        return to_repr(self)
