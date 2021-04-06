# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import copy
import time
import pandas as pd
from hypernets.conf import configure, Configurable, String, Int as cfg_int
from hypernets.core import TrialHistory, Trial, EarlyStoppingError
from hypernets.core.ops import Identity, HyperInput
from hypernets.core.search_space import HyperSpace, ParameterSpace
from hypernets.searchers import EvolutionSearcher, RandomSearcher, MCTSSearcher, GridSearcher, get_searcher_cls, \
    Searcher
from hypernets.utils import logging
from hypernets.utils.common import isnotebook
from IPython.display import clear_output, display, update_display

logger = logging.get_logger(__name__)


@configure()
class ParamSearchCfg(Configurable):
    work_dir = String(help='storage directory path to store running data.').tag(config=True)
    trial_retry_limit = cfg_int(1000, min=1, help='maximum retry number to run trial.').tag(config=True)


def func_space(func):
    space = HyperSpace()
    with space.as_default():
        params = {name: copy.copy(v) for name, v in zip(func.__code__.co_varnames, func.__defaults__) if
                  isinstance(v, ParameterSpace)}
        for _, v in params.items():
            v.attach_to_space(space, v.name)
        input = HyperInput()
        id1 = Identity(**params)(input)
        space.set_outputs([id1])
    return space


def build_searcher(cls, func, optimize_direction='min'):
    cls = get_searcher_cls(cls)
    search_space_fn = lambda: func_space(func)

    if cls == EvolutionSearcher:
        s = cls(search_space_fn, optimize_direction=optimize_direction,
                population_size=30, sample_size=10, candidates_size=10,
                regularized=True, use_meta_learner=True)
    elif cls == MCTSSearcher:
        s = MCTSSearcher(search_space_fn, optimize_direction=optimize_direction, max_node_space=10)
    elif cls == RandomSearcher:
        s = cls(search_space_fn, optimize_direction=optimize_direction)
    elif cls == GridSearcher:
        s = cls(search_space_fn, optimize_direction=optimize_direction)
    else:
        s = cls(search_space_fn, optimize_direction=optimize_direction)
    return s


def search_params(func, searcher='Grid', max_trials=100, optimize_direction='min', clear_logs=False, **func_kwargs):
    if not isinstance(searcher, Searcher):
        searcher = build_searcher(searcher, func, optimize_direction)
    retry_limit = ParamSearchCfg.trial_retry_limit
    trial_no = 1
    retry_counter = 0
    history = TrialHistory(optimize_direction)

    current_trial_display_id = None
    trials_display_id = None
    while trial_no <= max_trials:
        try:
            space_sample = searcher.sample()

            if isnotebook():
                if clear_logs:
                    clear_output()
                    current_trial_display_id = None
                    trials_display_id = None
                    
                if current_trial_display_id is None:
                    display({'text/markdown': '#### Current Trial:'}, raw=True, include=['text/markdown'])
                    handle = display(space_sample, display_id=True)
                    if handle is not None:
                        current_trial_display_id = handle.display_id
                else:
                    update_display(space_sample, display_id=current_trial_display_id)
                df_best_trials = pd.DataFrame([
                    (t.trial_no, t.reward, t.elapsed, t.space_sample.vectors) for t in history.get_top(100)],
                    columns=['Trial No.', 'Reward', 'Elapsed', 'Space Vector'])
                if trials_display_id is None:
                    display({'text/markdown': '#### Top trials:'}, raw=True, include=['text/markdown'])
                    handle = display(df_best_trials, display_id=True)
                    if handle is not None:
                        trials_display_id = handle.display_id
                else:
                    update_display(df_best_trials, display_id=trials_display_id)

            if history.is_existed(space_sample):
                if retry_counter >= retry_limit:
                    logger.info(f'Unable to take valid sample and exceed the retry limit {retry_limit}.')
                    break
                retry_counter += 1
                continue

            ps = space_sample.get_assigned_params()
            params = {p.alias.split('.')[-1]: p.value for p in ps}
            func_params = func_kwargs.copy()
            func_params.update(params)

            trial_start = time.time()
            last_reward = func(**func_params)
            elapsed = time.time() - trial_start

            trial = Trial(space_sample, trial_no, last_reward, elapsed)
            if last_reward != 0:  # success
                improved = history.append(trial)
            if logger.is_info_enabled():
                best = history.get_best()
                msg = f'Trial {trial_no} done, reward: {trial.reward}, ' \
                      f'best_trial_no:{best.trial_no}, best_reward:{best.reward}\n'
                logger.info(msg)
        except EarlyStoppingError:
            break
            # TODO: early stopping
        except Exception as e:
            import sys
            import traceback
            msg = f'{">" * 20} Trial {trial_no} failed! {"<" * 20}\n' \
                  + f'{e.__class__.__name__}: {e}\n' \
                  + traceback.format_exc() \
                  + '*' * 50
            logger.error(msg)
        finally:
            trial_no += 1
            retry_counter = 0
    return history
