# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.core.search_space import *
from hypernets.core.ops import *
import numpy as np
import pytest


class Test_Parameter:
    def test_alias(self):
        space = HyperSpace()
        with space.as_default():
            int1 = Int(1, 100)
            int2 = Int(1, 2)
            c1 = Choice([1, 2])

            assert int1.alias is None
            assert int2.alias is None
            assert c1.alias is None

            id1 = Identity(p1=int1, p2=int2, p3=c1)
            assert int1.alias == 'Module_Identity_1.p1'
            assert int2.alias == 'Module_Identity_1.p2'
            assert c1.alias == 'Module_Identity_1.p3'

            id2 = Identity(name = 'id2', action=int2)
            assert int2.alias == 'Module_Identity_1.p2,id2.action'

    def test_choice(self):
        state = np.random.RandomState(9527)
        c1 = Choice(['a', 'b', 'c', 'd'], random_state=state)

        v1 = c1.random_sample()
        assert v1 == c1.value
        assert v1 == 'a'

        c2 = Choice([1, 2, 3, 4, 5])
        c2.assign(1)
        assert c2.value == 1

        c3 = Choice([1, 'a', 'b', 4])
        with pytest.raises(AssertionError) as excinfo:
            c3.assign('f')

    def test_int(self):
        state = np.random.RandomState(9527)
        i1 = Int(0, 100, random_state=state)

        v1 = i1.random_sample()
        assert v1 == i1.value
        assert v1 == 97

        i2 = Int(0, 100)
        i2.assign(0)
        assert i2.value == 0

        i3 = Int(0, 100)
        i3.assign(100)
        assert i3.value == 100

        i4 = Int(0, 100)
        with pytest.raises(AssertionError) as excinfo:
            i4.assign(101)

    def test_real(self):
        state = np.random.RandomState(9527)
        r1 = Real(0, 1, prior="uniform", random_state=state)
        r2 = Real(0, 1, prior="q_uniform", q=0.2, random_state=state)
        r3 = Real(0.01, 1, prior="log_uniform", random_state=state)
        r4 = Real(0, 1)
        r5 = Real(0, 1)
        assert r1.assigned == False
        assert r4.assigned == False
        v1 = r1.random_sample()
        v2 = r2.random_sample()
        v3 = r3.random_sample()
        r4.assign(0.2)
        assert r1.assigned == True
        assert r4.assigned == True
        assert v1 == 0.33068165175773345
        assert v2 == 0.4
        assert v3 == 1.2745819909588392
        assert r4.value == 0.2

        with pytest.raises(AssertionError) as excinfo:
            r5.assign(2.0)

    def test_multiple_choice(self):
        mc1 = MultipleChoice(['a', 'b', 'c', 'd', 'e', 'f'], random_state=np.random.RandomState(1))
        mc2 = MultipleChoice(['a', 'b', 'c', 'd', 'e', 'f'], random_state=np.random.RandomState(2))
        mc3 = MultipleChoice([1, 2, 3, 4, 5], max_chosen_num=2, random_state=np.random.RandomState(3))
        mc4 = MultipleChoice(['a', 1, True, 'test', 5.6], max_chosen_num=1, random_state=np.random.RandomState(4))

        v1 = mc1.random_sample()
        v2 = mc2.random_sample()
        v3 = mc3.random_sample()
        v4 = mc4.random_sample()

        assert v1 == ['d', 'f']
        assert v2 == ['e', 'd']
        assert v3 == [4, 5]
        assert v4 == ['a']

        with pytest.raises(AssertionError):
            MultipleChoice([1])

        mc5 = MultipleChoice([1, 2, 3, 4, 5], max_chosen_num=2, random_state=np.random.RandomState(3))
        with pytest.raises(AssertionError):
            mc5.assign([])

    def test_dynamic(self):
        i1 = Int(1, 100)
        r1 = Real(0, 10.0)
        c1 = Choice(['a', 'b', 'c'])

        d1 = Dynamic(
            lambda args: f'i1:{args["i1"]}, r1:{args["r1"]}, c1:{args["c1"]}',
            i1=i1, r1=r1, c1=c1
        )

        i1.assign(33)
        assert d1.value is None
        assert d1.assigned == False

        r1.assign(0.99)
        assert d1.value is None
        assert d1.assigned == False

        c1.assign('b')
        assert d1.assigned == True
        assert d1.value == 'i1:33, r1:0.99, c1:b'

        d2 = Dynamic(
            lambda args: 'no dependent'
        )

        assert d2.assigned == True
        assert d2.value == 'no dependent'
