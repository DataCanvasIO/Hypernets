# -*- coding:utf-8 -*-
"""

"""
from hypernets.core.search_space import *
from hypernets.core.ops import *
import pytest


class Test_Module:
    def test_identity(self):
        id = Identity()
        id.compile([1, 2, 3])
        assert id.output, [1, 2, 3]

    def test_parameters(self):
        id = Identity()
        id.add_parameters(p1=Int(1, 10), p2=Choice([1, 2, 3]), p3=5)
        assert len(id._hyper_params) == 3
        assert id._hyper_params['p3'].value == 5

        id2 = Identity(p1=Int(1, 10), p2=Choice([1, 2, 3]), p3=5)
        assert len(id2._hyper_params) == 3
        assert id2._hyper_params['p3'].value == 5

        v1 = Int(1, 10)
        v2 = Choice([1, 2, 3])
        v3 = 5
        dict = {'p1': v1, 'p2': v2, 'p3': v3}
        id3 = Identity(**dict)
        assert len(id3._hyper_params) == 3
        assert id3._hyper_params['p3'].value == 5

        assert id3.is_params_ready == False
        assert id3.param_values == {'p1': None, 'p2': None, 'p3': 5}

        v1.assign(7)
        v2.assign(2)
        assert id3.is_params_ready == True
        assert id3.param_values == {'p1': 7, 'p2': 2, 'p3': 5}

    def test_optional(self):
        def get_space(keep_link=True):
            space = HyperSpace()
            with space.as_default():
                id1 = Identity()
                hp_opt = Bool()
                opt = Optional(Identity(), keep_link=keep_link, hp_opt=hp_opt)
                id2 = Identity()
                opt(id1)
                id2(opt)
            return space

        space = get_space()
        ids = []

        def get_id(m):
            ids.append(m.id)
            return True

        space.traverse(get_id)
        assert ids == ['Module_Identity_1', 'Module_Optional_1', 'Module_Identity_3']

        space.Param_Bool_1.assign(True)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Identity_1', 'Module_Identity_2', 'Module_Identity_3']

        space = get_space(keep_link=True)
        space.Param_Bool_1.assign(False)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Identity_1', 'Module_Identity_3']

        space = get_space(keep_link=False)
        space.Param_Bool_1.assign(False)
        ids = []
        with pytest.raises(ValueError) as excinfo:
            space.traverse(get_id)
        assert excinfo.value.args[0] == 'Graph is not connected.'

    def test_sequential(self):
        def get_space(keep_link=True):
            space = HyperSpace()
            with space.as_default():
                in1 = HyperInput()
                id1 = Identity()
                id2 = Identity()
                seq = Sequential([id1, id2])(in1)
                id3 = Identity()(seq)
                # id3(seq(in1))

            return space

        space = get_space()
        ids = []

        def get_id(m):
            ids.append(m.id)
            return True

        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Sequential_1', 'Module_Identity_3']

        ids = []
        space.random_sample()
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Identity_1', 'Module_Identity_2', 'Module_Identity_3']

    def test_or(self):
        def get_space(keep_link=True):
            space = HyperSpace()
            with space.as_default():
                in1 = HyperInput()
                id1 = Identity()
                id2 = Identity()
                orop = Or([id1, id2])
                id3 = Identity()
                orop(in1)
                id3(orop)

            return space

        space = get_space()
        ids = []

        def get_id(m):
            ids.append(m.id)
            return True

        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Or_1', 'Module_Identity_3']

        space.Module_Or_1.hp_or.assign(0)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Identity_1', 'Module_Identity_3']

        space = get_space()
        space.Module_Or_1.hp_or.assign(1)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Identity_2', 'Module_Identity_3']

    def test_input_choice(self):
        def get_space(keep_link=True):
            space = HyperSpace()
            with space.as_default():
                in1 = HyperInput()
                id1 = Identity()(in1)
                id2 = Identity()(in1)
                id3 = Identity()(in1)
                ic1 = InputChoice(3, 2)([id1, id2, id3])
                id4 = Identity()(ic1)
            return space

        space = get_space()
        ids = []

        def get_id(m):
            ids.append(m.id)
            return True

        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Identity_1', 'Module_Identity_2', 'Module_Identity_3',
                       'Module_InputChoice_1', 'Module_Identity_4']

        space.Module_InputChoice_1.hp_choice.assign([0, 1])
        assert len(space.edges - set([(space.Module_HyperInput_1, space.Module_Identity_2),
                               (space.Module_HyperInput_1, space.Module_Identity_3),
                               (space.Module_Identity_2, space.Module_Identity_4),
                               (space.Module_Identity_1, space.Module_Identity_4),
                               (space.Module_HyperInput_1, space.Module_Identity_1)]))==0

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([1])
        assert len(space.edges - set([(space.Module_HyperInput_1, space.Module_Identity_2),
                               (space.Module_HyperInput_1, space.Module_Identity_3),
                               (space.Module_Identity_2, space.Module_Identity_4),
                               (space.Module_HyperInput_1, space.Module_Identity_1)]))==0

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([2])
        assert len(space.edges - set([(space.Module_HyperInput_1, space.Module_Identity_2),
                               (space.Module_HyperInput_1, space.Module_Identity_3),
                               (space.Module_Identity_3, space.Module_Identity_4),
                               (space.Module_HyperInput_1, space.Module_Identity_1)]))==0