#
import copy
import pickle
from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from hypernets.core import set_random_state, randint
from hypernets.core.ops import ModuleChoice, HyperInput, ModuleSpace
from hypernets.core.search_space import HyperSpace, Choice, Int, Real, Cascade, Constant, HyperNode
from hypernets.model import Estimator, HyperModel
from hypernets.tabular import get_tool_box, column_selector
from hypernets.utils import fs, const


from hypernets.core import randint
from hypernets.core.ops import ModuleChoice, HyperInput
from hypernets.core.search_space import HyperSpace, Choice, Int, Real
from hypernets.pipeline.base import DataFrameMapper
from hypernets.pipeline.transformers import FeatureImportanceSelection

from hypernets.utils import logging



logger = logging.get_logger(__name__)


class PlainSearchSpace(object):
    def __init__(self, enable_dt=True, enable_lr=True, enable_nn=True, enable_dtr=False):
        assert enable_dt or enable_lr or enable_nn or enable_dtr

        super(PlainSearchSpace, self).__init__()

        self.enable_dt = enable_dt
        self.enable_dtr = enable_dtr
        self.enable_lr = enable_lr
        self.enable_nn = enable_nn

    # DecisionTreeClassifier
    @property
    def dt(self):
        return dict(
            cls=DecisionTreeClassifier,
            criterion=Choice(["gini", "entropy"]),
            splitter=Choice(["best", "random"]),
            max_depth=Choice([None, 3, 5, 10, 20, 50]),
            random_state=randint(),
        )

    @property
    def dtr(self):
        return dict(
            cls=DecisionTreeRegressor,
            splitter=Choice(["best", "random"]),
            max_depth=Choice([None, 3, 5, 10, 20, 50]),
            random_state=randint(),
        )

    # NN
    @property
    def nn(self):
        solver = Choice(['lbfgs', 'sgd', 'adam'])
        return dict(
            cls=MLPClassifier,
            max_iter=Int(500, 5000, step=500),
            activation=Choice(['identity', 'logistic', 'tanh', 'relu']),
            solver=solver,
            learning_rate=Choice(['constant', 'invscaling', 'adaptive']),
            learning_rate_init_stub=Cascade(partial(self._cascade, self._nn_learning_rate_init, 'slvr'), slvr=solver),
            random_state=randint(),
        )

    @staticmethod
    def _nn_learning_rate_init(slvr):
        if slvr in ['sgd' or 'adam']:
            return 'learning_rate_init', Choice([0.001, 0.01])
        else:
            return 'learning_rate_init', Constant(0.001)

    # LogisticRegression
    @property
    def lr(self):
        iters = [1000]
        while iters[-1] < 9000:
            iters.append(int(round(iters[-1] * 1.25, -2)))

        solver = Choice(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
        penalty = Cascade(partial(self._cascade, self._lr_penalty_fn, 'slvr'), slvr=solver)
        l1_ratio = Cascade(partial(self._cascade, self._lr_l1_ratio, 'penalty'), penalty=penalty)

        return dict(
            cls=LogisticRegression,
            max_iter=Choice(iters),
            solver=solver,
            penalty_stub=penalty,
            l1_ratio_stub=l1_ratio,
            random_state=randint(),
        )

    @staticmethod
    def _lr_penalty_fn(slvr):
        if slvr == 'saga':
            return 'penalty', Choice(['l2', 'elasticnet', 'l1', 'none'])
        else:
            return 'penalty', Constant('l2')

    @staticmethod
    def _lr_l1_ratio(penalty):
        if penalty in ['elasticnet', ]:
            return 'l1_ratio', Real(0.0, 1.0, step=0.1)
        else:
            return 'l1_ratio', Constant(None)

    # commons
    @staticmethod
    def _cascade(fn, key, args, space):
        with space.as_default():
            kvalue = args[key]
            if isinstance(kvalue, HyperNode):
                kvalue = kvalue.value
            return fn(kvalue)

    def create_feature_selection(self, hyper_input, importances, seq_no=0):
        from hypernets.pipeline.base import Pipeline

        selection = FeatureImportanceSelection(name=f'feature_importance_selection_{seq_no}',
                                               importances=importances,
                                               quantile=Real(0, 1, step=0.1))
        pipeline = Pipeline([selection],
                            name=f'feature_selection_{seq_no}',
                            columns=column_selector.column_all)(hyper_input)

        preprocessor = DataFrameMapper(default=False, input_df=True, df_out=True,
                                       df_out_dtype_transforms=None)([pipeline])

        return preprocessor

    # HyperSpace
    def __call__(self, *args, **kwargs):
        space = HyperSpace()

        with space.as_default():
            hyper_input = HyperInput(name='input1')

            estimators = []
            if self.enable_dt:
                estimators.append(self.dt)
            if self.enable_dtr:
                estimators.append(self.dtr)
            if self.enable_lr:
                estimators.append(self.lr)
            if self.enable_nn:
                estimators.append(self.nn)
            modules = [ModuleSpace(name=f'{e["cls"].__name__}', **e) for e in estimators]

            if "importances" in kwargs and kwargs["importances"] is not None:
                importances = kwargs.pop("importances")
                ss = self.create_feature_selection(hyper_input, importances)
                outputs = ModuleChoice(modules)(ss)
            else:
                outputs = ModuleChoice(modules)(hyper_input)
            space.set_inputs(hyper_input)

        return space


class PlainEstimator(Estimator):
    def __init__(self, space_sample, task=const.TASK_BINARY, transformer=None):
        assert task in {const.TASK_BINARY, const.TASK_MULTICLASS, const.TASK_REGRESSION}

        super(PlainEstimator, self).__init__(space_sample, task)

        # space, _ = space_sample.compile_and_forward()
        out = space_sample.get_outputs()[0]
        kwargs = out.param_values
        kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, HyperNode)}

        cls = kwargs.pop('cls')
        logger.info(f'create estimator {cls.__name__}: {kwargs}')
        self.model = cls(**kwargs)
        self.cls = cls
        self.model_args = kwargs
        self.transformer = transformer

        # fitted
        self.classes_ = None
        # self.cv_models_ = []

    def summary(self):
        pass

    def fit(self, X, y, **kwargs):
        eval_set = kwargs.pop('eval_set', None)  # ignore

        if self.transformer is not None:
            logger.info('fit_transform data')
            X = self.transformer.fit_transform(X, y)

        logger.info('bring X,y to local')
        X, y = get_tool_box(X, y).to_local(X, y)

        logger.info('fit model')
        self.model.fit(X, y, **kwargs)
        self.classes_ = getattr(self.model, 'classes_', None)
        self.cv_ = False
        self.cv_models_ = None

        return self

    def fit_cross_validation(self, X, y, stratified=True, num_folds=3, shuffle=False, random_state=9527, metrics=None,
                             **kwargs):
        assert num_folds > 0
        assert isinstance(metrics, (list, tuple))

        eval_set = kwargs.pop('eval_set', None)  # ignore

        if self.transformer is not None:
            logger.info('fit_transform data')
            X = self.transformer.fit_transform(X, y)

        logger.info('bring X,y to local')
        tb_original = get_tool_box(X, y)
        X, y = tb_original.to_local(X, y)

        tb = get_tool_box(X, y)
        if stratified and self.task == const.TASK_BINARY:
            iterators = tb.statified_kfold(n_splits=num_folds, shuffle=True, random_state=random_state)
        else:
            iterators = tb.kfold(n_splits=num_folds, shuffle=True, random_state=random_state)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values

        oof_ = None
        oof_scores = []
        cv_models = []
        x_vals = []
        y_vals = []
        X_trains = []
        y_trains = []
        logger.info('start training')
        for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X, y)):
            x_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
            x_val_fold, y_val_fold = X.iloc[valid_idx], y[valid_idx]

            logger.info(f'fit fold {n_fold}')
            fold_model = copy.deepcopy(self.model)
            fold_model.fit(x_train_fold, y_train_fold, **kwargs)

            # calc fold oof and score
            logger.info(f'calc fold {n_fold} score')
            if self.task == const.TASK_REGRESSION:
                proba = fold_model.predict(x_val_fold)
                preds = proba
            else:
                proba = fold_model.predict_proba(x_val_fold)
                if self.task == const.TASK_BINARY:
                    proba = tb.fix_binary_predict_proba_result(proba)

                proba_threshold = 0.5
                if proba.shape[-1] > 2:  # multiclass
                    preds = proba.argmax(axis=-1)
                else:  # binary:
                    preds = (proba[:, 1] > proba_threshold).astype('int32')
                preds = np.array(fold_model.classes_).take(preds, axis=0)

            if oof_ is None:
                if len(proba.shape) == 1:
                    oof_ = np.full(y.shape, np.nan, proba.dtype)
                else:
                    oof_ = np.full((y.shape[0], proba.shape[-1]), np.nan, proba.dtype)
            fold_scores = tb.metrics.calc_score(y_val_fold, preds, proba, metrics, task=self.task)

            # save fold result
            oof_[valid_idx] = proba
            oof_scores.append(fold_scores)
            cv_models.append(fold_model)

            x_vals.append(x_val_fold)
            y_vals.append(y_val_fold)
            X_trains.append(x_train_fold)
            y_trains.append(y_train_fold)

        self.classes_ = getattr(cv_models[0], 'classes_', None)
        self.cv_ = True
        self.cv_models_ = cv_models

        # calc final score with mean
        scores = pd.concat([pd.Series(s) for s in oof_scores], axis=1).mean(axis=1).to_dict()
        logger.info(f'fit_cross_validation score:{scores}, folds score:{oof_scores}')

        # return
        oof_, = tb_original.from_local(oof_)
        return scores, oof_, oof_scores, X_trains, y_trains, x_vals, y_vals

    def predict(self, X, **kwargs):
        eval_set = kwargs.pop('eval_set', None)  # ignore

        if self.transformer is not None:
            logger.info('transform local')
            X = self.transformer.transform(X)

        logger.info('bring X,y to local')
        tb_original = get_tool_box(X)
        X, = tb_original.to_local(X)

        if self.cv_:
            if self.task == const.TASK_REGRESSION:
                pred_sum = None
                for n, est in enumerate(self.cv_models_):
                    logger.info(f'predict estimator {n}')
                    pred = est.predict(X, **kwargs)
                    if pred_sum is None:
                        pred_sum = pred
                    else:
                        pred_sum += pred
                preds = pred_sum / len(self.cv_models_)
            else:
                logger.info('predict_proba')
                proba = self.predict_proba(X, ingore_transformer=True, **kwargs)

                logger.info('proba2predict')
                preds = self.proba2predict(proba)
                preds = np.array(self.classes_).take(preds, axis=0)
        else:
            logger.info('predict')
            preds = self.model.predict(X, **kwargs)

        preds, = tb_original.from_local(preds)
        return preds

    def predict_proba(self, X, *, ingore_transformer=False, **kwargs):
        eval_set = kwargs.pop('eval_set', None)  # ignore

        if not ingore_transformer and self.transformer is not None:
            logger.info('transform data')
            X = self.transformer.transform(X)

        tb_original = get_tool_box(X)
        X, = tb_original.to_local(X)

        tb = get_tool_box(X)
        if self.cv_models_:
            proba_sum = None
            for n, est in enumerate(self.cv_models_):
                logger.info(f'predict_proba estimator {n}')
                proba = est.predict_proba(X, **kwargs)
                if self.task == const.TASK_BINARY:
                    proba = tb.fix_binary_predict_proba_result(proba)
                if proba_sum is None:
                    proba_sum = proba
                else:
                    proba_sum += proba
            proba = proba_sum / len(self.cv_models_)
        else:
            logger.info('predict_proba')
            proba = self.model.predict_proba(X, **kwargs)
            if self.task == const.TASK_BINARY:
                proba = tb.fix_binary_predict_proba_result(proba)

        proba, = tb_original.from_local(proba)
        return proba

    def evaluate(self, X, y, metrics=None, **kwargs):
        if metrics is None:
            metrics = ['rmse'] if self.task == const.TASK_REGRESSION else ['accuracy']

        if self.task == const.TASK_REGRESSION:
            proba = None
            preds = self.predict(X, **kwargs)
        else:
            proba = self.predict_proba(X, **kwargs)
            preds = self.proba2predict(proba, proba_threshold=kwargs.get('proba_threshold', 0.5))

        scores = get_tool_box(y).metrics.calc_score(y, preds, proba, metrics, self.task)
        return scores

    def proba2predict(self, proba, proba_threshold=0.5):
        if self.task == const.TASK_REGRESSION:
            return proba

        logger.info('proba2predict')
        if proba.shape[-1] > 2:
            predict = proba.argmax(axis=-1)
        elif proba.shape[-1] == 2:
            predict = (proba[:, 1] > proba_threshold).astype('int32')
        else:
            predict = (proba > proba_threshold).astype('int32')
        if self.classes_ is not None:
            predict = get_tool_box(predict).take_array(self.classes_, predict, axis=0)
        return predict

    def save(self, model_file):
        with fs.open(model_file, 'wb') as f:
            pickle.dump(self, f, protocol=4)

    @staticmethod
    def load(model_file):
        with fs.open(model_file, 'rb') as f:
            return pickle.load(f)

    def get_iteration_scores(self):
        return []

    def __repr__(self):
        if self.cv_models_:
            return f'{self.__class__.__name__}:{self.cv_models_}'
        else:
            return f'{self.__class__.__name__}:{self.model}'


class PlainModel(HyperModel):
    def __init__(self, searcher, dispatcher=None, callbacks=None, reward_metric=None, task=None,
                 discriminator=None, transformer=None):
        super(PlainModel, self).__init__(searcher, dispatcher=dispatcher, callbacks=callbacks,
                                         reward_metric=reward_metric, task=task)
        self.transformer = transformer

    def _get_estimator(self, space_sample):
        if callable(self.transformer):
            transformer = self.transformer()
        else:
            transformer = self.transformer

        return PlainEstimator(space_sample, task=self.task, transformer=transformer)

    def load_estimator(self, model_file):
        return PlainEstimator.load(model_file)


def train(X_train, y_train, X_eval, y_eval, task=None, reward_metric=None, optimize_direction='max', **kwargs):
    from hypernets.core.callbacks import SummaryCallback
    from hypernets.searchers import make_searcher

    if task is None:
        task, _ = get_tool_box(y_train).infer_task_type(y_train)
    if reward_metric is None:
        reward_metric = 'rmse' if task == const.TASK_REGRESSION else 'accuracy'

    search_space = PlainSearchSpace()
    searcher = make_searcher('mcts', search_space, optimize_direction=optimize_direction)
    callbacks = [SummaryCallback()]
    hm = PlainModel(searcher=searcher, task=task, reward_metric=reward_metric, callbacks=callbacks)
    hm.search(X_train, y_train, X_eval, y_eval, **kwargs)
    best = hm.get_best_trial()
    model = hm.final_train(best.space_sample, X_train, y_train)
    return hm, model


def train_heart_disease(**kwargs):
    from hypernets.tabular.datasets import dsutils
    from sklearn.model_selection import train_test_split

    X = dsutils.load_heart_disease_uci()
    y = X.pop('target')

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=randint())
    X_train, X_eval, y_train, y_eval = \
        train_test_split(X_train, y_train, test_size=0.3, random_state=randint())

    kwargs = {'reward_metric': 'auc', 'max_trials': 10, **kwargs}
    hm, model = train(X_train, y_train, X_eval, y_eval, const.TASK_BINARY, **kwargs)

    print('-' * 50)
    scores = model.evaluate(X_test, y_test, metrics=['auc', 'accuracy', 'f1', 'recall', 'precision'])
    print('scores:', scores)

    trials = hm.get_top_trials(10)
    models = [hm.load_estimator(t.model_file) for t in trials]

    msgs = [f'{t.trial_no},{t.reward},{m.cls.__name__} {m.model_args}' for t, m in zip(trials, models)]
    print('top trials:')
    print('\n'.join(msgs))


if __name__ == '__main__':
    set_random_state(335)
    train_heart_disease()
