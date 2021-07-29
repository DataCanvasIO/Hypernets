# -*- coding:utf-8 -*-
"""

"""
from sklearn.metrics import get_scorer

from hypernets.discriminators import make_discriminator
from hypernets.experiment import CompeteExperiment
from hypernets.experiment.cfg import ExperimentCfg as cfg
from hypernets.model import HyperModel
from hypernets.searchers import make_searcher, PlaybackSearcher
from hypernets.tabular import dask_ex as dex
from hypernets.tabular.cache import clear as _clear_cache
from hypernets.tabular.metrics import metric_to_scoring
from hypernets.utils import const, load_data, infer_task_type, hash_data, logging, isnotebook, load_module, DocLens

logger = logging.get_logger(__name__)


def make_experiment(hyper_model_cls,
                    train_data,
                    target=None,
                    eval_data=None,
                    test_data=None,
                    task=None,
                    id=None,
                    callbacks=None,
                    searcher=None,
                    searcher_options=None,
                    search_space=None,
                    search_callbacks=None,
                    early_stopping_rounds=10,
                    early_stopping_time_limit=3600,
                    early_stopping_reward=None,
                    reward_metric=None,
                    optimize_direction=None,
                    hyper_model_options=None,
                    clear_cache=None,
                    discriminator=None,
                    log_level=None,
                    **kwargs):
    """
    Utility to make CompeteExperiment instance with specified settings.

    Parameters
    ----------
    hyper_model_cls: subclass of HyperModel
        Subclass of HyperModel to run trials within the experiment.
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
    searcher_options: dict, optional, default is None
        The options to create searcher, is used if searcher is str.
    search_space : callable, optional
        Used to initialize searcher instance (if searcher is None, str or class).
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
    hyper_model_options: dict, optional
        Options to initlize HyperModel except *reward_metric*, *task*, *callbacks*, *discriminator*.
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
            if col.lower() in cfg.experiment_default_target_set:
                return col
        raise ValueError(f'Not found one of {cfg.experiment_default_target_set} from your data,'
                         f' implicit target must be specified.')

    def default_searcher(cls, options):
        assert search_space is not None, '"search_space" should be specified when "searcher" is None or str.'
        assert optimize_direction in {'max', 'min'}
        if options is None:
            options = {}
        options['optimize_direction'] = optimize_direction
        s = make_searcher(cls, search_space, **options)

        return s

    def to_search_object(sch):
        from hypernets.core.searcher import Searcher as SearcherSpec
        from hypernets.searchers import EvolutionSearcher

        if sch is None:
            sch = default_searcher(EvolutionSearcher, searcher_options)
        elif isinstance(sch, (type, str)):
            sch = default_searcher(sch, searcher_options)
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
        dc_nan_chars = kwargs.get('data_cleaner_args', {}).get('nan_chars')
        if isinstance(dc_nan_chars, str):
            dc_nan_chars = [dc_nan_chars]
        task, _ = infer_task_type(y_train, excludes=dc_nan_chars if dc_nan_chars is not None else None)

    if reward_metric is None:
        reward_metric = 'rmse' if task == const.TASK_REGRESSION else 'accuracy'
        logger.info(f'no reward metric specified, use "{reward_metric}" for {task} task by default.')

    if kwargs.get('scorer') is None:
        scorer = metric_to_scoring(reward_metric, task=task, pos_label=kwargs.get('pos_label'))
    else:
        scorer = kwargs.pop('scorer')

    if isinstance(scorer, str):
        scorer = get_scorer(scorer)

    if optimize_direction is None or len(optimize_direction) == 0:
        optimize_direction = 'max' if scorer._sign > 0 else 'min'

    searcher = to_search_object(searcher)

    if cfg.experiment_auto_down_sample_enabled and not isinstance(searcher, PlaybackSearcher) \
            and 'down_sample_search' not in kwargs.keys():
        train_data_shape = dex.compute(X_train.shape)[0] if dex.is_dask_object(X_train) else X_train.shape
        if train_data_shape[0] > cfg.experiment_auto_down_sample_rows_threshold:
            kwargs['down_sample_search'] = True

    if search_callbacks is None:
        search_callbacks = default_search_callbacks()
    search_callbacks = append_early_stopping_callbacks(search_callbacks)

    if callbacks is None:
        callbacks = default_experiment_callbacks()

    if discriminator is None and cfg.experiment_discriminator is not None and len(cfg.experiment_discriminator) > 0:
        discriminator = make_discriminator(cfg.experiment_discriminator,
                                           optimize_direction=optimize_direction,
                                           **(cfg.experiment_discriminator_options or {}))

    if id is None:
        id = hash_data(dict(X_train=X_train, y_train=y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval,
                            eval_size=kwargs.get('eval_size'), target=target, task=task))
        id = f'{hyper_model_cls.__name__}_{id}'

    if hyper_model_options is None:
        hyper_model_options = {}
    hm = hyper_model_cls(searcher, reward_metric=reward_metric, task=task, callbacks=search_callbacks,
                         discriminator=discriminator, **hyper_model_options)

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


def _merge_doc():
    my_doc = DocLens(make_experiment.__doc__)
    exp_doc = DocLens(CompeteExperiment.__init__.__doc__)
    excluded = ['hyper_model', 'X_train', 'y_train', 'X_eval', 'y_eval', 'X_test']
    params = my_doc.merge_parameters(exp_doc, exclude=excluded)
    for k in ['clear_cache', 'log_level']:
        params.move_to_end(k)
    params.pop('kwargs')
    my_doc.parameters = params

    make_experiment.__doc__ = my_doc.render()


_merge_doc()
