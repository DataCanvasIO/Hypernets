# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import pytest

from hypernets.core.ops import *
from hypernets.core.search_space import *
from hypernets.searchers import GridSearcher
from hypernets.core import EarlyStoppingError


def get_space():
    space = HyperSpace()
    state = np.random.RandomState(9527)

    with space.as_default():
        id1 = Identity(p1=Choice(['a', 'b'], random_state=state),
                       p2=Int(1, 100, random_state=state),
                       p3=Real(0, 1.0, random_state=state))
    return space


class Test_GridSearcher():
    def test_playback_searcher(self):
        searcher = GridSearcher(get_space, n_expansion=2)
        assert searcher.grid == {'Param_Choice_1': ['a', 'b'], 'Param_Int_1': [9, 98],
                                 'Param_Real_1': [0.23, 0.95]}

        assert searcher.all_combinations == [{'Param_Choice_1': 'a', 'Param_Int_1': 9, 'Param_Real_1': 0.23},
                                             {'Param_Choice_1': 'a', 'Param_Int_1': 9,
                                              'Param_Real_1': 0.95},
                                             {'Param_Choice_1': 'a', 'Param_Int_1': 98, 'Param_Real_1': 0.23},
                                             {'Param_Choice_1': 'a', 'Param_Int_1': 98,
                                              'Param_Real_1': 0.95},
                                             {'Param_Choice_1': 'b', 'Param_Int_1': 9, 'Param_Real_1': 0.23},
                                             {'Param_Choice_1': 'b', 'Param_Int_1': 9,
                                              'Param_Real_1': 0.95},
                                             {'Param_Choice_1': 'b', 'Param_Int_1': 98, 'Param_Real_1': 0.23},
                                             {'Param_Choice_1': 'b', 'Param_Int_1': 98,
                                              'Param_Real_1': 0.95}]
        sample1 = searcher.sample()
        assert sample1.vectors == [0, 9, 0.23]
        sample2 = searcher.sample()
        assert sample2.vectors == [0, 9, 0.95]
        sample3 = searcher.sample()
        assert sample3.vectors == [0, 98, 0.23]
        sample4 = searcher.sample()
        assert sample4.vectors == [0, 98, 0.95]
        sample5 = searcher.sample()
        assert sample5.vectors == [1, 9, 0.23]
        sample6 = searcher.sample()
        assert sample6.vectors == [1, 9, 0.95]
        sample7 = searcher.sample()
        assert sample7.vectors == [1, 98, 0.23]
        sample8 = searcher.sample()
        assert sample8.vectors == [1, 98, 0.95]

        with pytest.raises(EarlyStoppingError) as ese:
            searcher.sample()
        assert ese.value.args[0] == 'no more samples.'
