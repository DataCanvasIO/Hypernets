# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import pytest

from hypernets.core.ops import *
from hypernets.core.search_space import *
from hypernets.searchers import PlaybackSearcher
from hypernets.core import TrialHistory, Trial
from hypernets.core import EarlyStoppingError

def get_space():
    space = HyperSpace()
    with space.as_default():
        id1 = Identity(p1=Choice(['a', 'b']), p2=Int(1, 100), p3=Real(0, 1.0))
    return space


th = TrialHistory('min')
sample = get_space()
sample.assign_by_vectors([0, 1, 0.1])
trial = Trial(sample, 1, 0.99, 100)
th.append(trial)

sample = get_space()
sample.assign_by_vectors([1, 2, 0.2])
trial = Trial(sample, 2, 0.9, 50)
th.append(trial)

sample = get_space()
sample.assign_by_vectors([0, 3, 0.3])
trial = Trial(sample, 3, 0.7, 200)
th.append(trial)


class Test_PlaybackSearcher():
    def test_playback_searcher(self):
        searcher = PlaybackSearcher(th, top_n=2)
        sample1 = searcher.sample()
        assert sample1.vectors == [0, 3, 0.3]
        sample2 = searcher.sample()
        assert sample2.vectors == [1, 2, 0.2]
        with pytest.raises(EarlyStoppingError) as ese:
            searcher.sample()
        assert ese.value.args[0] == 'no more samples.'
