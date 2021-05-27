# -*- coding:utf-8 -*-
"""

"""

from sklearn.metrics import get_scorer

from hypernets.experiment import CompeteExperiment
from hypernets.experiment.cfg import ExperimentCfg as cfg
from hypernets.model import HyperModel
from hypernets.searchers import make_searcher
from hypernets.tabular import dask_ex as dex
from hypernets.tabular.cache import clear as _clear_cache
from hypernets.tabular.metrics import metric_to_scoring
from hypernets.utils import load_data, infer_task_type, hash_data, logging, const, isnotebook, load_module

logger = logging.get_logger(__name__)

DEFAULT_TARGET_SET = {'y', 'target'}


def make_experiment(hyper_model_cls,
                    train_data,
                    target=None,
                    eval_data=None,
                    test_data=None,
                    task=None,
                    id=None,
                    callbacks=None,
                    searcher=None,
                    search_space=None,
                    search_callbacks=None,
                    early_stopping_rounds=10,
                    early_stopping_time_limit=3600,
                    early_stopping_reward=None,
                    reward_metric=None,
                    optimize_direction=None,
                    clear_cache=None,
                    discriminator=None,
                    log_level=None,
                    **kwargs):
    """
    Ulitily to make CompeteExperiment instance with specified settings.

    Parameters
    ----------
    hyper_model_cls: subclass of HyperModel
        Subclass of HyperModel to run trials within the experiemnt.
    train_data : str, Pandas or Dask DataFrame
        Feature data for training with target column.
        For str, it's should be the data path in file system,
        we'll detect data format from this path (only .csv and .parquet are supported now) and read it.
    target : str, optional
        Target feature name for training, which must be one of the drain_data columns, default is 'y'.
    eval_data : str, Pandas or Dask DataFrame, optional
        Feature data for evaluation with target column.
        For str, it's should be the data path in file system,
        we'll detect data format from this path (only .csv and .parquet are supported now) and read it.
    test_data : str, Pandas or Dask DataFrame, optional
        Feature data for testing without target column.
        For str, it's should be the data path in file system,
        we'll detect data format from this path (only .csv and .parquet are supported now) and read it.
    task : str or None, (default=None)
        Task type(*binary*, *multiclass* or *regression*).
        If None, inference the type of task automatically
    id : str or None, (default=None)
        The experiment id.
    callbacks: list of ExperimentCallback, optional
        ExperimentCallback list.
    searcher : str, searcher class, search object, optional
        The hypernets Searcher instance to explore search space, default is EvolutionSearcher instance.
        For str, should be one of 'evolution', 'mcts', 'random'.
        For class, should be one of EvolutionSearcher, MCTSSearcher, RandomSearcher, or subclass of hypernets Searcher.
        For other, should be instance of hypernets Searcher.
    search_space : callable, optional
        Used to initialize searcher instance (if searcher is None, str or class),
        default is hypergbm.search_space.search_space_general (if Dask isn't enabled)
        or hypergbm.dask.search_space.search_space_general (if Dask is enabled) .
    search_callbacks
        Hypernets search callbacks, used to initialize searcher instance (if searcher is None, str or class).
        If log_level >= WARNNING, default is EarlyStoppingCallback only.
        If log_level < WARNNING, defalult is EarlyStoppingCallback plus SummaryCallback.
    early_stopping_rounds :　int, optional
        Setting of EarlyStoppingCallback, is used if EarlyStoppingCallback instance not found from search_callbacks.
        Set zero or None to disable it, default is 10.
    early_stopping_time_limit : int, optional
        Setting of EarlyStoppingCallback, is used if EarlyStoppingCallback instance not found from search_callbacks.
        Set zero or None to disable it, default is 3600 seconds.
    early_stopping_reward : float, optional
        Setting of EarlyStoppingCallback, is used if EarlyStoppingCallback instance not found from search_callbacks.
        Set zero or None to disable it, default is None.
    reward_metric : str, callable, optional, (default 'accuracy' for binary/multicalss task, 'rmse' for regression task)
        Hypernets search reward metric name or callable. Possible values:
            - accuracy
            - auc
            - f1
            - logloss
            - mse
            - mae
            - msle
            - precision
            - rmse
            - r2
            - recall
    optimize_direction : str, optional
        Hypernets search reward metric direction, default is detected from reward_metric.
    discriminator : instance of hypernets.discriminator.BaseDiscriminator, optional
        Discriminator is used to determine whether to continue training
    clear_cache: bool, optional, (default False)
    log_level : int, str, or None, (default=None),
        Level of logging, possible values:
            -logging.CRITICAL
            -logging.FATAL
            -logging.ERROR
            -logging.WARNING
            -logging.WARN
            -logging.INFO
            -logging.DEBUG
            -logging.NOTSET
    kwargs:
        Parameters to initialize experiment instance, refrence CompeteExperiment for more details.
    Returns
    -------
    Runnable experiment object

    Examples:
    -------
    Create experiment with example PlainModel and PlainSearchSpace:
    >>> from hypernets.examples.plain_model import PlainModel, PlainSearchSpace
    >>> from hypernets.experiment import make_experiment
    >>> from hypernets.tabular.datasets import dsutils
    >>> df = dsutils.load_blood()
    >>> experiment = make_experiment(PlainModel, df, target='Class', search_space=PlainSearchSpace())
    >>> estimator = experiment.run()

    """
    assert callable(hyper_model_cls) and hasattr(hyper_model_cls, '__name__')
    assert train_data is not None, 'train data is required.'
    if not issubclass(hyper_model_cls, HyperModel):
        logger.warning(f'{hyper_model_cls.__name__} isn\'t subclass of HyperModel.')

    kwargs = kwargs.copy()

    if log_level is None:
        log_level = logging.WARN
    logging.set_level(log_level)

    def find_target(df):
        columns = df.columns.to_list()
        for col in columns:
            if col.lower() in DEFAULT_TARGET_SET:
                return col
        raise ValueError(f'Not found one of {DEFAULT_TARGET_SET} from your data, implicit target must be specified.')

    def default_searcher(cls):
        assert search_space is not None, '"search_space" should be specified when "searcher" is None or str.'
        op = optimize_direction if optimize_direction is not None \
            else 'max' if scorer._sign > 0 else 'min'

        s = make_searcher(cls, search_space, optimize_direction=op)

        return s

    def to_search_object(sch):
        from hypernets.core.searcher import Searcher as SearcherSpec
        from hypernets.searchers import EvolutionSearcher

        if sch is None:
            sch = default_searcher(EvolutionSearcher)
        elif isinstance(sch, (type, str)):
            sch = default_searcher(sch)
        elif not isinstance(sch, SearcherSpec):
            logger.warning(f'Unrecognized searcher "{sch}".')

        return sch

    def default_experiment_callbacks():
        cbs = cfg.experiment_callbacks_notebook if isnotebook() else cfg.experiment_callbacks_console
        cbs = [load_module(cb)() if isinstance(cb, str) else cb for cb in cbs]
        return cbs

    def default_search_callbacks():
        cbs = cfg.hyper_model_callbacks_notebook if isnotebook() else cfg.hyper_model_callbacks_console
        cbs = [load_module(cb)() if isinstance(cb, str) else cb for cb in cbs]
        return cbs

    def append_early_stopping_callbacks(cbs):
        from hypernets.core.callbacks import EarlyStoppingCallback

        assert isinstance(cbs, (tuple, list))
        if any([isinstance(cb, EarlyStoppingCallback) for cb in cbs]):
            return cbs

        op = optimize_direction if optimize_direction is not None \
            else 'max' if scorer._sign > 0 else 'min'
        es = EarlyStoppingCallback(early_stopping_rounds, op,
                                   time_limit=early_stopping_time_limit,
                                   expected_reward=early_stopping_reward)

        return [es] + cbs

    X_train, X_eval, X_test = [load_data(data) if data is not None else None
                               for data in (train_data, eval_data, test_data)]

    X_train, X_eval, X_test = [dex.reset_index(x) if dex.is_dask_dataframe(x) else x
                               for x in (X_train, X_eval, X_test)]

    if target is None:
        target = find_target(X_train)

    y_train = X_train.pop(target)
    y_eval = X_eval.pop(target) if X_eval is not None else None

    if task is None:
        task, _ = infer_task_type(y_train)

    if reward_metric is None:
        reward_metric = 'rmse' if task == const.TASK_REGRESSION else 'accuracy'
        logger.info(f'no reward metric specified, use "{reward_metric}" for {task} task by default.')

    scorer = metric_to_scoring(reward_metric) if kwargs.get('scorer') is None else kwargs.pop('scorer')

    if isinstance(scorer, str):
        scorer = get_scorer(scorer)

    searcher = to_search_object(searcher)

    if search_callbacks is None:
        search_callbacks = default_search_callbacks()
    search_callbacks = append_early_stopping_callbacks(search_callbacks)

    if callbacks is None:
        callbacks = default_experiment_callbacks()

    if id is None:
        id = hash_data(dict(X_train=X_train, y_train=y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval,
                            eval_size=kwargs.get('eval_size'), target=target, task=task))
        id = f'{hyper_model_cls.__name__}_{id}'

    hm = hyper_model_cls(searcher, reward_metric=reward_metric, callbacks=search_callbacks,
                         discriminator=discriminator)

    experiment = CompeteExperiment(hm, X_train, y_train, X_eval=X_eval, y_eval=y_eval, X_test=X_test,
                                   task=task, id=id, callbacks=callbacks, scorer=scorer, **kwargs)

    if clear_cache:
        _clear_cache()

    if logger.is_info_enabled():
        train_shape, test_shape, eval_shape = \
            dex.compute(X_train.shape,
                        X_test.shape if X_test is not None else None,
                        X_eval.shape if X_eval is not None else None,
                        traverse=True)
        logger.info(f'make_experiment with train data:{train_shape}, '
                    f'test data:{test_shape}, eval data:{eval_shape}, target:{target}')

    return experiment
