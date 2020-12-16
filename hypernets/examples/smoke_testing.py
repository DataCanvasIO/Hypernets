# -*- coding:utf-8 -*-
"""

"""
import numpy as np

from hypernets.core.ops import Choice, Bool, Identity
from hypernets.core.search_space import HyperSpace, Int, Real
from hypernets.searchers.evolution_searcher import EvolutionSearcher
from hypernets.searchers.mcts_searcher import MCTSSearcher
from hypernets.searchers.random_searcher import RandomSearcher


def get_space():
    space = HyperSpace()
    with space.as_default():
        p1 = Int(1, 100)
        p2 = Choice(['a', 'b', 'c'])
        p3 = Bool()
        p4 = Real(0.0, 1.0)
        id1 = Identity(p1=p1)
        id2 = Identity(p2=p2)(id1)
        id3 = Identity(p3=p3)(id2)
        id4 = Identity(p4=p4)(id3)
    return space


def run_search():
    searchers = (
        RandomSearcher(get_space, space_sample_validation_fn=lambda s: True),
        MCTSSearcher(get_space, max_node_space=10),
        EvolutionSearcher(get_space, 5, 3, regularized=False)
    )

    for searcher in searchers:
        for i in range(100):
            space_sample = searcher.sample()
            assert space_sample.all_assigned == True
            print(searcher.__class__.__name__, i, space_sample.params_summary())
            searcher.update_result(space_sample, np.random.uniform(0.1, 0.9))


if __name__ == '__main__':
    run_search()
