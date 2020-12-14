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

            id2 = Identity(name='id2', action=int2)
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
        assert v1 == 0.33
        assert v2 == 0.4
        assert v3 == 1.2700501670841682
        assert r4.value == 0.2

        with pytest.raises(AssertionError) as excinfo:
            r5.assign(2.0)

    def test_multiple_choice(self):
        mc1 = MultipleChoice(['a', 'b', 'c', 'd', 'e', 'f'], random_state=np.random.RandomState(1))
        mc2 = MultipleChoice(['a', 'b', 'c', 'd', 'e', 'f'], random_state=np.random.RandomState(2))
        mc3 = MultipleChoice([1, 2, 3, 4, 5], num_chosen_most=2, random_state=np.random.RandomState(3))
        mc4 = MultipleChoice(['a', 1, True, 'test', 5.6], num_chosen_most=1, random_state=np.random.RandomState(4))

        v1 = mc1.random_sample()
        v2 = mc2.random_sample()
        v3 = mc3.random_sample()
        v4 = mc4.random_sample()

        assert v1 == ['a', 'b', 'c', 'd', 'e', 'f']
        assert v2 == ['e']
        assert v3 == [4]
        assert v4 == ['a']

        with pytest.raises(AssertionError):
            MultipleChoice([1],num_chosen_least=2)

        mc5 = MultipleChoice([1, 2, 3, 4, 5], num_chosen_most=2, random_state=np.random.RandomState(3))
        with pytest.raises(AssertionError):
            mc5.assign([])

    def test_dynamic(self):
        i = Int(1, 100)
        r = Real(0, 10.0)
        c = Choice(['a', 'b', 'c'])

        d1 = Dynamic(lambda i1, r1, c1: f'i1:{i1}, r1:{r1}, c1:{c1}', i1=i, r1=r, c1=c)

        i.assign(33)
        assert d1.value is None
        assert d1.assigned == False

        r.assign(0.99)
        assert d1.value is None
        assert d1.assigned == False

        c.assign('b')
        assert d1.assigned == True
        assert d1.value == 'i1:33, r1:0.99, c1:b'

        d2 = Dynamic(
            lambda: 'no dependent'
        )

        assert d2.assigned == True
        assert d2.value == 'no dependent'

    def test_cascade(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                c1 = Choice(['a', 'b', 'c'])

                def cc1_fn(c1, s):
                    with s.as_default():
                        if c1 == 'a':
                            return 'm1', Choice([1, 2])
                        elif c1 == 'b':
                            return 'm1', Choice([3, 4])
                        else:
                            return 'm1', Choice([5, 6])

                cc1 = Cascade(lambda args, space: cc1_fn(args['c1'], space), c1=c1)

                def cc2_fn(m1, s):
                    with s.as_default():
                        if isinstance(m1, ParameterSpace):
                            m1 = m1.value
                        if m1 == 5:
                            return 'm2', Choice([11, 22])
                        elif m1 == 6:
                            return 'm2', Choice([33, 44])
                        else:
                            return 'm2', Constant(-1)

                cc2 = Cascade(lambda args, space: cc2_fn(args['m1'], space), m1=cc1)

                def cc3_fn(m2, s):
                    with s.as_default():
                        if isinstance(m2, ParameterSpace):
                            m2 = m2.value
                        if m2 == 11:
                            return 'm3', Choice([55, 66])
                        else:
                            return 'm3', Constant(-1)

                cc3 = Cascade(lambda args, space: cc3_fn(args['m2'], space), m2=cc2)
                id1 = Identity(p1=c1, p2=cc1, p3=cc2, p4=cc3)
            return space

        space = get_space()
        space.Param_Choice_1.assign('a')
        assert space.Param_Cascade_1.value.options == [1, 2]

        space = get_space()
        space.Param_Choice_1.assign('b')
        assert space.Param_Cascade_1.value.options == [3, 4]

        space = get_space()
        space.Param_Choice_1.assign('c')
        assert space.Param_Cascade_1.value.options == [5, 6]

        assert space.Module_Identity_1.all_assigned == False
        space.Param_Choice_2.assign(5)

        assert space.Param_Cascade_2.value.options == [11, 22]

        assert space.Module_Identity_1.all_assigned == False
        space.Param_Choice_3.assign(22)

        assert space.Param_Cascade_3.value.value == -1
        assert space.Module_Identity_1.all_assigned == True

        assert len(space.get_assigned_params()) == 3

    def test_same(self):
        int1 = Int(1, 10)
        int2 = Int(1, 10)
        assert int1.same_config(int2)
        int2.alias = 'int2'
        assert not int1.same_config(int2)

        int3 = Int(0, 1)
        assert not int1.same_config(int3)

        real1 = Real(0, 1)
        real2 = Real(0, 1)

        assert real1.same_config(real2)

        real3 = Real(0, 1, q=1)
        assert not real1.same_config(real3)

        choice1 = Choice([1, 2])
        choice2 = Choice([1, 2])
        assert choice1.same_config(choice2)

        choice3 = Choice([1, 2, 3])
        assert not choice1.same_config(choice3)

        bool1 = Bool()
        bool2 = Bool()
        assert bool1.same_config(bool2)

        const1 = Constant(1)
        const2 = Constant(1)
        assert const1.same_config(const2)
        const3 = Constant('b')
        assert not const1.same_config(const3)

    def test_numeric2value(self):
        int1 = Int(1, 100)
        assert int1.value2numeric(10) == 10
        assert int1.numeric2value(10) == 10

        real1 = Real(0., 100.)
        assert real1.value2numeric(10.) == 10.
        assert real1.numeric2value(10.) == 10.

        bool1 = Bool()
        assert bool1.value2numeric(True) == 1
        assert bool1.numeric2value(1) == True

        choice1 = Choice(['a', 'b', 'c', 'd'])
        assert choice1.value2numeric('c') == 2
        assert choice1.numeric2value(2) == 'c'

        mutiple_choice1 = MultipleChoice(['a', 'b', 'c', 'd'])
        assert mutiple_choice1.value2numeric(['b', 'd']) == 5
        assert mutiple_choice1.numeric2value(5) == ['b', 'd']

    def test_label(self):
        space = HyperSpace()
        with space.as_default():
            int1 = Int(1, 100)
            assert int1.label == 'Param_Int_1-1-100-1'
            real1 = Real(0., 100.)
            assert real1.label == 'Param_Real_1-0.0-100.0-None-uniform-0.01'
            bool1 = Bool()
            assert bool1.label == 'Param_Bool_1-[False, True]'

            choice1 = Choice(['a', 'b', 'c', 'd'])
            assert choice1.label == 'Param_Choice_1-[\'a\', \'b\', \'c\', \'d\']'
            mutiple_choice1 = MultipleChoice(['a', 'b', 'c', 'd'])
            assert mutiple_choice1.label == 'Param_MultipleChoice_1-[\'a\', \'b\', \'c\', \'d\']-0-1'

    def test_expansion(self):
        int1 = Int(1, 100)
        vs = int1.expansion(20)
        assert len(vs) == 20
        vs = int1.expansion(101)
        assert len(vs) == 99
        vs = int1.expansion(0)
        assert len(vs) == 99

        real1 = Real(0., 100., max_expansion=10)
        vs = real1.expansion(20)
        assert len(vs) == 20
        vs = real1.expansion(0)
        assert len(vs) == 10

        bool1 = Bool()
        assert len(bool1.expansion()) == 2

        choice1 = Choice(['a', 'b', 'c', 'd'])
        assert len(choice1.expansion()) == 4

        mutiple_choice1 = MultipleChoice(['a', 'b', 'c', 'd'])
        vs = mutiple_choice1.expansion(20)
        assert len(vs) == 15

        vs = mutiple_choice1.expansion(0)
        assert len(vs) == 15

        mutiple_choice2 = MultipleChoice(['a', 'b', 'c', 'd'], num_chosen_most=1)
        vs = mutiple_choice2.expansion(4)
        assert len(vs) == 4
        vs = mutiple_choice2.expansion(20)
        assert len(vs) == 4
