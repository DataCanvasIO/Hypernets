# -*- coding:utf-8 -*-
"""

"""
__author__ = 'yangjian'

from .evolution_searcher import EvolutionSearcher
from .mcts_searcher import MCTSSearcher
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
    'Playback': PlaybackSearcher
}


def get_searcher_cls(identifier):
    from ..core.searcher import Searcher

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
