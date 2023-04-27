# -*- coding:utf-8 -*-
"""

"""
from sklearn.metrics import get_scorer

from hypernets.discriminators import make_discriminator
from hypernets.experiment import CompeteExperiment
from hypernets.experiment.cfg import ExperimentCfg as cfg
from hypernets.model import HyperModel
from hypernets.model.objectives import create_objective
from hypernets.searchers import make_searcher, PlaybackSearcher
from hypernets.tabular import get_tool_box
from hypernets.tabular.cache import clear as _clear_cache
from hypernets.tabular.cfg import TabularCfg as tcfg
from hypernets.utils import const, logging, isnotebook, load_module, DocLens

logger = logging.get_logger(__name__)


def find_target(df):
    columns = df.columns.to_list()
    for col in columns:
        if col.lower() in cfg.experiment_default_target_set:
            return col
    raise ValueError(f'Not found one of {cfg.experiment_default_target_set} from your data,'
                     f' implicit target must be specified.')


def default_experiment_callbacks():
    cbs = cfg.experiment_callbacks_notebook if isnotebook() else cfg.experiment_callbacks_console
    cbs = [load_module(cb)() if isinstance(cb, str) else cb for cb in cbs]
    return cbs


def default_search_callbacks():
    cbs = cfg.hyper_model_callbacks_notebook if isnotebook() else cfg.hyper_model_callbacks_console
    cbs = [load_module(cb)() if isinstance(cb, str) else cb for cb in cbs]
    return cbs


def to_objective_object(o, force_minimize=False, **kwargs):
    from hypernets.core.objective import Objective

    if isinstance(o, str):
        return create_objective(o, force_minimize=force_minimize, **kwargs)
    elif isinstance(o, Objective):
        return o
    else:
        raise RuntimeError("objective specific should be instanced by 'Objective' or a string")


def to_search_object(search_space, optimize_direction, searcher, searcher_options,
                     reward_metric=None, scorer=None, objectives=None,  task=None, pos_label=None):

    def to_searcher(cls, options):
        assert search_space is not None, '"search_space" should be specified if "searcher" is None or str.'
        assert optimize_direction in {'max', 'min'}
        s = make_searcher(cls, search_space, optimize_direction=optimize_direction, **options)

        return s

    if searcher is None:
        from hypernets.searchers import EvolutionSearcher
        sch = to_searcher(EvolutionSearcher, searcher_options)
    elif isinstance(searcher, (type, str)):
        from hypernets.searchers.moo import MOOSearcher
        from hypernets.searchers import get_searcher_cls

        search_cls = get_searcher_cls(searcher)
        if issubclass(search_cls, MOOSearcher):
            from hypernets.model.objectives import PredictionObjective
            from hypernets.searchers.moead_searcher import MOEADSearcher
            from hypernets.core import get_random_state

            if objectives is None:
                objectives = ['nf']
            objectives_instance = []
            force_minimize = (search_cls == MOEADSearcher)
            for o in objectives:
                objectives_instance.append(to_objective_object(o, force_minimize=force_minimize,
                                                               task=task, pos_label=pos_label))

            objectives_instance.insert(0, PredictionObjective.create(reward_metric, force_minimize=force_minimize,
                                                                     task=task, pos_label=pos_label))
            searcher_options['objectives'] = objectives_instance
            searcher_options['random_state'] = get_random_state()

        sch = to_searcher(searcher, searcher_options)
    else:
        from hypernets.core.searcher import Searcher as SearcherSpec
        if not isinstance(searcher, SearcherSpec):
            logger.warning(f'Unrecognized searcher "{searcher}".')
        sch = searcher

    return sch


def to_report_render_object(render, options):
    from hypernets.experiment.report import ReportRender, get_render
    options = {} if options is None else options
    if isinstance(render, ReportRender):
        return render
    elif isinstance(render, str):
        return get_render(render)(**options)
    else:
        raise ValueError(f"Unknown report render '{render}' ")


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
                    objectives=None,
                    optimize_direction=None,
                    hyper_model_options=None,
                    discriminator=None,
                    evaluation_metrics='auto',
                    evaluation_persist_prediction=False,
                    evaluation_persist_prediction_dir=None,
                    report_render=None,
                    report_render_options=None,
                    experiment_cls=None,
                    n_jobs=None,
                    clear_cache=None,
                    log_level=None,
                    **kwargs):
    """
    Utility to make CompeteExperiment instance with specified settings.

    Parameters
    ----------
    hyper_model_cls: subclass of HyperModel
        Subclass of HyperModel to run trials within the experiment.
    train_data : str, Pandas or Dask or Cudf DataFrame
        Feature data for training with target column.
        For str, it's should be the data path in file system, will be loaded as pnadas Dataframe.
        we'll detect data format from this path (only .csv and .parquet are supported now).
    target : str, optional
        Target feature name for training, which must be one of the drain_data columns, default is 'y'.
    eval_data : str, Pandas or Dask or Cudf DataFrame, optional
        Feature data for evaluation, should be None or have the same python type with 'train_data'.
    test_data : str, Pandas or Dask or Cudf DataFrame, optional
        Feature data for testing without target column, should be None or have the same python type with 'train_data'.
    task : str or None, (default=None)
        Task type(*binary*, *multiclass* or *regression*).
        If None, inference the type of task automatically
    id : str or None, (default=None)
        The experiment id.
    callbacks: list of ExperimentCallback, optional
        ExperimentCallback list.
    searcher : str, searcher class, search object, optional
        The hypernets Searcher instance to explore search space, default is EvolutionSearcher instance.
        For str, should be one of 'evolution', 'mcts', 'random', 'nsga2', 'moead'.  # TODO rnsga
        For class, should be one of EvolutionSearcher, MCTSSearcher, RandomSearcher, MOEADSearcher, NSGAIISearrcher
         or subclass of hypernets Searcher.
        For other, should be instanced of hypernets Searcher.
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
    reward_metric : str, callable, optional, (default 'accuracy' for binary/multiclass task, 'rmse' for regression task)
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
    objectives : List[Union[Objective, str]] optional, (default to ['nf'] )
        Used for multi-objectives optimization, "reward_metric" is alway picked as the first objective, specilly for
        "MOEADSearcher", will force the indicator to be the smaller the better by converting score to a negative number.
        For str as identifier of objectives, possible values:
            - elapsed
            - pred_perf
            - nf
    optimize_direction : str, optional
        Hypernets search reward metric direction, default is detected from reward_metric.
    discriminator : instance of hypernets.discriminator.BaseDiscriminator, optional
        Discriminator is used to determine whether to continue training
    hyper_model_options: dict, optional
        Options to initlize HyperModel except *reward_metric*, *task*, *callbacks*, *discriminator*.
    evaluation_metrics: str, list, or None (default='auto'),
        If *eval_data* is not None, it used to evaluate model with the metrics.
        For str should be 'auto', it will select metrics accord to machine learning task type.
        For list should be metrics name.
    evaluation_persist_prediction: bool (default=False)
    evaluation_persist_prediction_dir: str or None (default='predction')
        The dir to persist prediction, if exists will be overwritten
    report_render: str, obj, optional, default is None
        The experiment report render.
        For str should be one of 'excel'
        for obj should be instanced of ReportRender
    report_render_options: dict, optional
        The options to create render, is used if render is str.
    experiment_cls: class, or None, (default=CompeteExperiment)
        The experiment type, CompeteExperiment or it's subclass.
    n_jobs: int, default None
        If not None, update value of option `TabularCfg.joblib_njobs`.
    clear_cache: bool, optional, (default False)
        Clear cache store before running the expeirment.
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
    assert train_data is not None, 'train_data is required.'
    assert eval_data is None or type(eval_data) is type(train_data)
    assert test_data is None or type(test_data) is type(train_data)
    assert n_jobs is None or isinstance(n_jobs, int)

    if not issubclass(hyper_model_cls, HyperModel):
        logger.warning(f'{hyper_model_cls.__name__} isn\'t subclass of HyperModel.')

    kwargs = kwargs.copy()

    if log_level is None:
        log_level = logging.WARN
    logging.set_level(log_level)

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

    if isinstance(train_data, str):
        import pandas as pd
        tb = get_tool_box(pd.DataFrame)
        X_train = tb.load_data(train_data, reset_index=True)
        X_eval = tb.load_data(eval_data, reset_index=True) if eval_data is not None else None
        X_test = tb.load_data(test_data, reset_index=True) if test_data is not None else None
    else:
        tb = get_tool_box(train_data, eval_data, test_data)
        X_train = tb.reset_index(train_data)
        X_eval = tb.reset_index(eval_data) if eval_data is not None else None
        X_test = tb.reset_index(test_data) if test_data is not None else None

    if target is None:
        target = find_target(X_train)

    y_train = X_train.pop(target)
    y_eval = X_eval.pop(target) if X_eval is not None else None

    if task is None:
        dc_nan_chars = kwargs.get('data_cleaner_args', {}).get('nan_chars')
        if isinstance(dc_nan_chars, str):
            dc_nan_chars = [dc_nan_chars]
        task, _ = tb.infer_task_type(y_train, excludes=dc_nan_chars if dc_nan_chars is not None else None)

    if reward_metric is None:
        reward_metric = 'rmse' if task == const.TASK_REGRESSION else 'accuracy'
        logger.info(f'no reward metric specified, use "{reward_metric}" for {task} task by default.')

    if kwargs.get('scorer') is None:
        scorer = tb.metrics.metric_to_scoring(reward_metric, task=task, pos_label=kwargs.get('pos_label'))
    else:
        scorer = kwargs.pop('scorer')

    if isinstance(scorer, str):
        scorer = get_scorer(scorer)

    if optimize_direction is None or len(optimize_direction) == 0:
        optimize_direction = 'max' if scorer._sign > 0 else 'min'

    if searcher_options is None:
        searcher_options = {}

    searcher = to_search_object(search_space, optimize_direction, searcher, searcher_options,
                                reward_metric=reward_metric, scorer=scorer, objectives=objectives, task=task,
                                pos_label=kwargs.get('pos_label'))

    if searcher.kind() == const.SEARCHER_MOO:
        if 'psi' in [_.name for _ in searcher.objectives]:
            assert X_test is not None, "psi objective requires test dataset"

    if cfg.experiment_auto_down_sample_enabled and not isinstance(searcher, PlaybackSearcher) \
            and 'down_sample_search' not in kwargs.keys():
        train_data_shape = tb.get_shape(X_train)
        if train_data_shape[0] > cfg.experiment_auto_down_sample_rows_threshold:
            kwargs['down_sample_search'] = True

    if search_callbacks is None:
        search_callbacks = default_search_callbacks()
    search_callbacks = append_early_stopping_callbacks(search_callbacks)

    if callbacks is None:
        callbacks = default_experiment_callbacks()

    if eval_data is not None:
        from hypernets.experiment import MLEvaluateCallback
        if task in [const.TASK_REGRESSION, const.TASK_BINARY, const.TASK_MULTICLASS]\
                and searcher.kind() == const.SEARCHER_SOO:
            if evaluation_persist_prediction is True:
                persist_dir = evaluation_persist_prediction_dir
            else:
                persist_dir = None
            callbacks.append(MLEvaluateCallback(evaluation_metrics, persist_dir))

    if report_render is not None:
        from hypernets.experiment import MLReportCallback
        report_render = to_report_render_object(report_render, report_render_options)
        callbacks.append(MLReportCallback(report_render))

    if discriminator is None and cfg.experiment_discriminator is not None and len(cfg.experiment_discriminator) > 0:
        discriminator = make_discriminator(cfg.experiment_discriminator,
                                           optimize_direction=optimize_direction,
                                           **(cfg.experiment_discriminator_options or {}))

    if id is None:
        hasher = tb.data_hasher()
        id = hasher(dict(X_train=X_train, y_train=y_train, X_test=X_test, X_eval=X_eval, y_eval=y_eval,
                         eval_size=kwargs.get('eval_size'), target=target, task=task))
        id = f'{hyper_model_cls.__name__}_{id}'

    if hyper_model_options is None:
        hyper_model_options = {}
    hm = hyper_model_cls(searcher, reward_metric=reward_metric, task=task, callbacks=search_callbacks,
                         discriminator=discriminator, **hyper_model_options)

    if n_jobs is not None:
        tcfg.joblib_njobs = n_jobs

    if experiment_cls is None:
        experiment_cls = CompeteExperiment
    experiment = experiment_cls(hm, X_train, y_train, X_eval=X_eval, y_eval=y_eval, X_test=X_test,
                                task=task, id=id, callbacks=callbacks, scorer=scorer, **kwargs)

    if clear_cache:
        _clear_cache()

    if logger.is_info_enabled():
        train_shape = tb.get_shape(X_train)
        test_shape = tb.get_shape(X_test, allow_none=True)
        eval_shape = tb.get_shape(X_eval, allow_none=True)
        logger.info(f'make_experiment with train data:{train_shape}, '
                    f'test data:{test_shape}, eval data:{eval_shape}, target:{target}, task:{task}')

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
