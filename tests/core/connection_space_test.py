# -*- coding:utf-8 -*-
"""

"""
from hypernets.core.search_space import *
from hypernets.core.ops import *
import pytest


def block1(step):
    id1 = Identity(name=f'{step}_block1_id_1')
    id2 = Identity(name=f'{step}_block1_id_2')(id1)
    return id2


def block2(step):
    id1_1 = Identity(name=f'{step}_block2_id_1_1')
    id1_2 = Identity(name=f'{step}_block2_id_1_2')
    id2 = Identity(name=f'{step}_block2_id_2')([id1_1, id1_2])
    return id2


def block3(step):
    id1 = Identity(name=f'{step}_block3_id_1')
    id2_1 = Identity(name=f'{step}_block2_id_2_1')(id1)
    id2_2 = Identity(name=f'{step}_block2_id_2_2')(id1)
    return [id2_1, id2_2]


ids = []


def get_id(m):
    ids.append(m.id)
    return True


class Test_ConnectionSpace:
    def test_identity(self):
        id = Identity()
        id.compile_and_forward([1,2,3])
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

        global ids
        space = get_space()
        ids = []
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

    def test_optional_subgraph(self):
        def get_space(keep_link=True):
            space = HyperSpace()
            with space.as_default():
                input = Identity(name='input')
                hp_opt = Bool()
                opt = Optional(block1(1), keep_link=keep_link, hp_opt=hp_opt)(input)
                output = Identity(name='output')(opt)
                space.set_inputs(input)
            return space

        space = get_space()
        global ids
        space.Param_Bool_1.assign(True)
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input', 'ID_1_block1_id_1', 'ID_1_block1_id_2', 'ID_output']

        space = get_space()
        space.Param_Bool_1.assign(False)
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input', 'ID_output']

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

        global ids
        space = get_space()
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Sequential_1', 'Module_Identity_3']

        ids = []
        space.random_sample()
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Identity_1', 'Module_Identity_2', 'Module_Identity_3']

    def test_sequential_subgraph(self):
        global ids
        space = HyperSpace()
        with space.as_default():
            seq = Sequential([block1(1), block1(2)])
            space.random_sample()
            ids = []
            space.traverse(get_id)
            assert ids == ['ID_1_block1_id_1', 'ID_1_block1_id_2', 'ID_2_block1_id_1', 'ID_2_block1_id_2']

        space = HyperSpace()
        with space.as_default():
            input = Identity(name='input')
            seq = Sequential([block1(1), block1(2)])(input)
            output = Identity(name='output')(seq)
            space.random_sample()
            ids = []
            space.traverse(get_id)
            assert ids == ['ID_input', 'ID_1_block1_id_1', 'ID_1_block1_id_2', 'ID_2_block1_id_1', 'ID_2_block1_id_2',
                           'ID_output']

        space = HyperSpace()
        with space.as_default():
            input = Identity(name='input')
            Sequential([block1(1), block2(1)])(input)
            ids = []
            space.random_sample()
            space.traverse(get_id)
            assert ids == ['ID_input', 'ID_1_block1_id_1', 'ID_1_block1_id_2', 'ID_1_block2_id_1_1',
                           'ID_1_block2_id_1_2',
                           'ID_1_block2_id_2']

        space = HyperSpace()
        with space.as_default():
            Sequential([block1(1), block2(1), block3(1)])
            ids = []
            space.random_sample()
            space.traverse(get_id)
            assert ids == ['ID_1_block1_id_1', 'ID_1_block1_id_2', 'ID_1_block2_id_1_1', 'ID_1_block2_id_1_2',
                           'ID_1_block2_id_2',
                           'ID_1_block3_id_1', 'ID_1_block2_id_2_1', 'ID_1_block2_id_2_2']

    def test_permutation(self):
        def get_space(keep_link=True):
            space = HyperSpace()
            with space.as_default():
                in1 = HyperInput()
                id1 = Identity()
                id2 = Identity()
                seq = Permutation([id1, id2])(in1)
                id3 = Identity()(seq)
                # id3(seq(in1))

            return space

        global ids

        space = get_space()
        ids = []

        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Permutation_1', 'Module_Identity_3']

        assert space.Param_Choice_1.options == [(0, 1), (1, 0)]
        ids = []
        space.Param_Choice_1.assign((1, 0))
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Identity_2', 'Module_Identity_1', 'Module_Identity_3']

    def test_or(self):
        def get_space(keep_link=True):
            space = HyperSpace()
            with space.as_default():
                in1 = HyperInput()
                id1 = Identity()
                id2 = Identity()
                orop = ModuleChoice([id1, id2])
                id3 = Identity()
                orop(in1)
                id3(orop)

            return space

        global ids
        space = get_space()
        ids = []

        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_ModuleChoice_1', 'Module_Identity_3']

        space.Module_ModuleChoice_1.hp_or.assign(0)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Identity_1', 'Module_Identity_3']

        space = get_space()
        space.Module_ModuleChoice_1.hp_or.assign(1)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Identity_2', 'Module_Identity_3']

    def test_or_subgraph(self):
        def get_space(keep_link=True):
            space = HyperSpace()
            with space.as_default():
                in1 = HyperInput(name='input')
                orop = ModuleChoice([block1(1), block2(1)])(in1)
                output = Identity(name='output')(orop)
                space.set_inputs(in1)
            return space

        global ids
        space = get_space()
        ids = []

        space.traverse(get_id)
        assert ids == ['ID_input', 'Module_ModuleChoice_1', 'ID_output']

        space.Module_ModuleChoice_1.hp_or.assign(0)
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input', 'ID_1_block1_id_1', 'ID_1_block1_id_2', 'ID_output']

        space = get_space()
        space.Module_ModuleChoice_1.hp_or.assign(1)
        ids = []
        space.traverse(get_id)
        assert ids == ['ID_input', 'ID_1_block2_id_1_1', 'ID_1_block2_id_1_2', 'ID_1_block2_id_2', 'ID_output']

    def test_repeat_id(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                in1 = HyperInput()
                rep = Repeat(module_fn=lambda step: Identity(p1=Choice([1, 2, 3])), repeat_times=[2, 3, 4])(in1)
                id3 = Identity()(rep)
            return space

        global ids
        space = get_space()
        ids = []

        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Repeat_1', 'Module_Identity_1']

        assert space.Param_Choice_1.options == [2, 3, 4]
        ids = []
        space.Param_Choice_1.assign(3)
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Identity_2', 'Module_Identity_3', 'Module_Identity_4',
                       'Module_Identity_1']
        assert len(space.hyper_params) == 4

    def test_repeat_seq(self):
        def seq_fn(step):
            id1 = Identity(p1=Choice([1, 2]))
            id2 = Identity(p2=Int(1, 10))
            seq = Sequential([id1, id2])
            return seq

        def get_space():
            space = HyperSpace()
            with space.as_default():
                in1 = HyperInput()
                rep = Repeat(module_fn=seq_fn, repeat_times=[2, 3, 4])(in1)
                id3 = Identity()(rep)
            return space

        space = get_space()
        global ids
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Repeat_1', 'Module_Identity_1']

        assert space.Param_Choice_1.options == [2, 3, 4]
        ids = []
        space.Param_Choice_1.assign(3)
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Sequential_1', 'Module_Sequential_2', 'Module_Sequential_3',
                       'Module_Identity_1']
        assert len(space.hyper_params) == 10

        space.random_sample()
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Identity_2', 'Module_Identity_3', 'Module_Identity_4',
                       'Module_Identity_5', 'Module_Identity_6', 'Module_Identity_7', 'Module_Identity_1']

    def test_repeat_subgraph(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                in1 = HyperInput()
                rep = Repeat(module_fn=block1, repeat_times=[2, 3, 4])(in1)
                id3 = Identity()(rep)
            return space

        space = get_space()
        global ids
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Repeat_1', 'Module_Identity_1']

        assert space.Param_Choice_1.options == [2, 3, 4]
        ids = []
        space.Param_Choice_1.assign(3)
        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'ID_0_block1_id_1', 'ID_0_block1_id_2', 'ID_1_block1_id_1',
                       'ID_1_block1_id_2', 'ID_2_block1_id_1', 'ID_2_block1_id_2', 'Module_Identity_1']

    def test_input_choice(self):
        def get_space(keep_link=True):
            space = HyperSpace()
            with space.as_default():
                in1 = HyperInput()
                id1 = Identity()(in1)
                id2 = Identity()(in1)
                id3 = Identity()(in1)
                ic1 = InputChoice([id1, id2, id3], 2)([id1, id2, id3])
                id4 = Identity()(ic1)
            return space

        global ids
        space = get_space()
        ids = []

        space.traverse(get_id)
        assert ids == ['Module_HyperInput_1', 'Module_Identity_1', 'Module_Identity_2', 'Module_Identity_3',
                       'Module_InputChoice_1', 'Module_Identity_4']

        space.Module_InputChoice_1.hp_choice.assign([0, 1])
        assert len(space.edges - set([(space.Module_HyperInput_1, space.Module_Identity_2),
                                      (space.Module_HyperInput_1, space.Module_Identity_3),
                                      (space.Module_Identity_2, space.Module_Identity_4),
                                      (space.Module_Identity_1, space.Module_Identity_4),
                                      (space.Module_HyperInput_1, space.Module_Identity_1)])) == 0

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([1])
        assert len(space.edges - set([(space.Module_HyperInput_1, space.Module_Identity_2),
                                      (space.Module_HyperInput_1, space.Module_Identity_3),
                                      (space.Module_Identity_2, space.Module_Identity_4),
                                      (space.Module_HyperInput_1, space.Module_Identity_1)])) == 0

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([2])
        assert len(space.edges - set([(space.Module_HyperInput_1, space.Module_Identity_2),
                                      (space.Module_HyperInput_1, space.Module_Identity_3),
                                      (space.Module_Identity_3, space.Module_Identity_4),
                                      (space.Module_HyperInput_1, space.Module_Identity_1)])) == 0
