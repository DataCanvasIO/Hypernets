# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import copy
import time

from hypernets.conf import configure, Configurable, String, Int as CfgInt
from hypernets.core import TrialHistory, Trial, EarlyStoppingError, EarlyStoppingCallback
from hypernets.core.ops import Identity, HyperInput
from hypernets.core.search_space import HyperSpace, ParameterSpace
from hypernets.searchers import EvolutionSearcher, RandomSearcher, MCTSSearcher, GridSearcher, get_searcher_cls, \
    Searcher
from hypernets.utils import logging

logger = logging.get_logger(__name__)


@configure()
class ParamSearchCfg(Configurable):
    work_dir = String(help='storage directory path to store running data.').tag(config=True)
    trial_retry_limit = CfgInt(1000, min=1, help='maximum retry number to run trial.').tag(config=True)


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


def search_params(func, searcher='Grid', max_trials=100, optimize_direction='min', history=None, callbacks=None,
                  **func_kwargs):
    if callbacks is not None:
        assert len(callbacks) == 1, "Only accept one EarlyStoppingCallback's instance."
        assert isinstance(callbacks[0], EarlyStoppingCallback), "Only accept EarlyStoppingCallback's instance yet."
    else:
        callbacks = []

    if not isinstance(searcher, Searcher):
        searcher = build_searcher(searcher, func, optimize_direction)

    for callback in callbacks:
        callback.on_search_start(None, None, None, None, None, None, None, max_trials, None, None)

    retry_limit = ParamSearchCfg.trial_retry_limit
    trial_no = 1
    retry_counter = 0
    if history is None:
        history = TrialHistory(optimize_direction)
    else:
        assert isinstance(history, TrialHistory)

    while trial_no <= max_trials:
        try:
            space_sample = searcher.sample()

            for callback in callbacks:
                callback.on_trial_begin(None, space_sample, trial_no)

            if history.is_existed(space_sample):
                trial = history.get_trial(space_sample)
                if trial is not None:
                    searcher.update_result(space_sample, trial.reward)
                if retry_counter >= retry_limit:
                    logger.info(f'Unable to take valid sample and exceed the retry limit {retry_limit}.')
                    break
                logger.info(f'skip trial {trial.trial_no}')
                retry_counter += 1
                continue

            ps = space_sample.get_assigned_params()
            params = {p.alias.split('.')[-1]: p.value for p in ps}
            func_params = func_kwargs.copy()
            func_params.update(params)

            trial_start = time.time()
            result = func(**func_params)
            assert isinstance(result, (float, dict, list))

            if isinstance(result, dict):
                assert 'reward' in result.keys()
                last_reward = [result['reward']]
            elif isinstance(result, float):
                last_reward = [result]
            else:
                last_reward = result
            searcher.update_result(space_sample, last_reward)
            elapsed = time.time() - trial_start

            trial = Trial(space_sample, trial_no, last_reward, elapsed)
            if isinstance(result, dict):
                memo = result.copy()
                memo.pop('reward')
                trial.memo.update(memo)

            if last_reward[0] != 0:  # success
                improved = history.append(trial)
                for callback in callbacks:
                    callback.on_trial_end(None, space_sample, trial_no, trial.reward, improved, trial.elapsed)
            if logger.is_info_enabled():
                best = history.get_best()
                msg = f'Trial {trial_no} done, reward: {trial.reward}, ' \
                      f'best_trial_no:{best.trial_no}, best_reward:{best.reward}\n'
                logger.info(msg)
        except EarlyStoppingError:
            break
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
