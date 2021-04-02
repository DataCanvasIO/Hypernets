#
import pickle
from functools import partial

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from hypernets.core.callbacks import SummaryCallback
from hypernets.core.ops import ModuleChoice, HyperInput, ModuleSpace
from hypernets.core.search_space import HyperSpace, Choice, Int, Real, Cascade, Constant, HyperNode
from hypernets.model import Estimator, HyperModel
from hypernets.searchers import RandomSearcher
from hypernets.tabular.metrics import calc_score
from hypernets.utils import fs, logging, const, infer_task_type

logger = logging.get_logger(__name__)


class PlainSearchSpace:
    # DecisionTreeClassifier
    @property
    def dt(self):
        return dict(
            cls=DecisionTreeClassifier,
            criterion=Choice(["gini", "entropy"]),
            splitter=Choice(["best", "random"]),
            max_depth=Choice([None, 3, 5, 10, 20, 50]),
        )

    # NN
    @property
    def nn(self):
        solver = Choice(['lbfgs', 'sgd', 'adam'])
        return dict(
            cls=MLPClassifier,
            max_iter=Int(1000, 10000, step=500),
            activation=Choice(['identity', 'logistic', 'tanh', 'relu']),
            solver=solver,
            learning_rate=Choice(['constant', 'invscaling', 'adaptive']),
            learning_rate_init_stub=Cascade(partial(self._cascade, self._nn_learning_rate_init, 'slvr'), slvr=solver)
        )

    @staticmethod
    def _nn_learning_rate_init(slvr):
        if slvr in ['sgd' or 'adam']:
            return 'learning_rate_init', Choice([0.0001, 0.001, 0.01])
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
            penalty_sub=penalty,
            l1_ratio_stub=l1_ratio,
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

    # HyperSpace
    def __call__(self, *args, **kwargs):
        space = HyperSpace()

        with space.as_default():
            hyper_input = HyperInput(name='input1')
            estimators = [self.dt, self.nn, self.lr]
            # estimators = [self.nn, ]
            modules = [ModuleSpace(name=f'{e["cls"].__name__}', **e) for e in estimators]
            outputs = ModuleChoice(modules)(hyper_input)
            space.set_inputs(hyper_input)

        return space


class PlainEstimator(Estimator):
    def __init__(self, space_sample, task=const.TASK_BINARY):
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
        self.classes_ = None

    def summary(self):
        pass

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        if hasattr(self.model, 'classes_'):
            self.classes_ = getattr(self.model, 'classes_')
        return self

    def fit_cross_validation(self, X, y, stratified=True, num_folds=3, shuffle=False, random_state=9527, metrics=None):
        raise NotImplemented()

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        return self.model.predict_proba(X, **kwargs)

    def evaluate(self, X, y, metrics=None, **kwargs):
        if metrics is None:
            metrics = ['rmse'] if self.task == const.TASK_REGRESSION else ['accuracy']

        if self.task != const.TASK_REGRESSION:
            proba = self.predict_proba(X, **kwargs)
        else:
            proba = None
        preds = self.predict(X, **kwargs)
        scores = calc_score(y, preds, proba, metrics, self.task)
        return scores

    def save(self, model_file):
        with fs.open(model_file, 'wb') as f:
            pickle.dump(self, f, protocol=4)

    @staticmethod
    def load(model_file):
        with fs.open(model_file, 'rb') as f:
            return pickle.load(f)


class PlainModel(HyperModel):
    def __init__(self, searcher, dispatcher=None, callbacks=None, reward_metric=None, task=None):
        super(PlainModel, self).__init__(searcher, dispatcher=dispatcher, callbacks=callbacks,
                                         reward_metric=reward_metric, task=task)

    def _get_estimator(self, space_sample):
        return PlainEstimator(space_sample, task=self.task)

    def load_estimator(self, model_file):
        return PlainEstimator.load(model_file)


def train(X_train, y_train, X_eval, y_eval, task=None, reward_metric=None, optimize_direction='max', max_trials=10):
    if task is None:
        task = infer_task_type(y_train)
    if reward_metric is None:
        reward_metric = 'rmse' if task == const.TASK_REGRESSION else 'accuracy'

    search_space = PlainSearchSpace()
    searcher = RandomSearcher(search_space, optimize_direction=optimize_direction)
    callbacks = [SummaryCallback()]
    hm = PlainModel(searcher=searcher, task=task, reward_metric=reward_metric, callbacks=callbacks)
    hm.search(X_train, y_train, X_eval, y_eval, cv=False, max_trials=max_trials)
    best = hm.get_best_trial()
    model = hm.final_train(best.space_sample, X_train, y_train)
    return hm, model


def train_heart_disease():
    from hypernets.tabular.datasets import dsutils
    from sklearn.model_selection import train_test_split

    X = dsutils.load_heart_disease_uci()
    y = X.pop('target')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.3)

    hm, model = train(X_train, y_train, X_eval, y_eval, const.TASK_BINARY, 'auc', max_trials=100)

    print('-' * 50)
    scores = model.evaluate(X_test, y_test, metrics=['auc', 'accuracy', 'f1', 'recall', 'precision'])
    print('scores:', scores)

    trials = hm.get_top_trials(10)
    models = [hm.load_estimator(t.model_file) for t in trials]

    msgs = [f'{t.trial_no},{t.reward},{m.cls.__name__} {m.model_args}' for t, m in zip(trials, models)]
    print('top trials:')
    print('\n'.join(msgs))


if __name__ == '__main__':
    train_heart_disease()
