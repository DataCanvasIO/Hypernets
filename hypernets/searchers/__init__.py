# -*- coding:utf-8 -*-
"""

"""
__author__ = 'yangjian'

from .evolution_searcher import EvolutionSearcher
from .mcts_searcher import MCTSSearcher
from .moead_searcher import MOEADSearcher
from .nsga_searcher import NSGAIISearcher, RNSGAIISearcher
from .random_searcher import RandomSearcher
from .playback_searcher import PlaybackSearcher
from .grid_searcher import GridSearcher
from ..core.searcher import Searcher

searcher_dict = {
    'mcts': MCTSSearcher,
    'MCTS': MCTSSearcher,
    'MCTSSearcher': MCTSSearcher,
    'evolution': EvolutionSearcher,
    'Evolution': EvolutionSearcher,
    'EvolutionSearcher': EvolutionSearcher,
    'Random': RandomSearcher,
    'RandomSearcher': RandomSearcher,
    'random': RandomSearcher,
    'grid': GridSearcher,
    'Grid': GridSearcher,
    'GridSearcher': GridSearcher,
    'playback': PlaybackSearcher,
    'PlaybackSearcher': PlaybackSearcher,
    'Playback': PlaybackSearcher,
    'nsga2': NSGAIISearcher,
    'rnsga2': RNSGAIISearcher,
    'NSGAIISearcher': NSGAIISearcher,
    'RNSGAIISearcher': RNSGAIISearcher,
    'moead': MOEADSearcher,
    'MOEADSearcher': MOEADSearcher
}


def get_searcher_cls(identifier):
    if isinstance(identifier, str):
        cls = searcher_dict.get(identifier, None)
        if cls is None:
            raise ValueError(f'Illegal identifier:{identifier}')
        else:
            return cls
    elif isinstance(identifier, type):
        if issubclass(identifier, Searcher):
            return identifier
        else:
            raise ValueError(f'Wrong searcher type:{identifier}')
    else:
        raise ValueError(f'Illegal identifier:{identifier}')


def make_searcher(cls, search_space_fn, optimize_direction='min', objectives=None, **kwargs):
    from hypernets.searchers.moo import MOOSearcher

    cls = get_searcher_cls(cls)

    if cls == EvolutionSearcher:
        default_kwargs = dict(population_size=30, sample_size=10, candidates_size=10,
                              regularized=True, use_meta_learner=True, optimize_direction=optimize_direction)
    elif cls == MCTSSearcher:
        default_kwargs = dict(max_node_space=10, optimize_direction=optimize_direction)
    elif issubclass(cls, MOOSearcher):
        default_kwargs = dict(objectives=objectives)
    else:
        default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    searcher = cls(search_space_fn, **kwargs)
    return searcher
