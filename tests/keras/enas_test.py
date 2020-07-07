# -*- coding:utf-8 -*-
"""

"""
from hypernets.frameworks.keras.enas_common_ops import sepconv2d_bn, sepconv3x3, sepconv5x5, avgpooling3x3, \
    maxpooling3x3, add, identity, conv_op, conv_node, conv_cell
from hypernets.frameworks.keras.layers import Input
from hypernets.core.ops import *
from hypernets.core.search_space import HyperSpace
from tensorflow.keras.utils import plot_model

ids = []


def get_id(m):
    ids.append(m.id)
    return True


class Test_Enas():
    def test_enas_ops(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                name_prefix = 'test_'
                filters = 64
                in1 = Input(shape=(28, 28, 1,))
                in2 = Input(shape=(28, 28, 1,))
                ic1 = InputChoice([in1, in2], 1)([in1, in2])
                or1 = Or([sepconv5x5(name_prefix, filters),
                          sepconv3x3(name_prefix, filters),
                          avgpooling3x3(name_prefix, filters),
                          maxpooling3x3(name_prefix, filters),
                          identity(name_prefix)])(ic1)
                space.set_inputs([in1, in2])
                return space

        space = get_space()
        global ids
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_Input_2', 'Module_InputChoice_1', 'Module_Or_1']

        assert len(space.edges & {(space.Module_Input_1, space.Module_InputChoice_1),
                                  (space.Module_Input_2, space.Module_InputChoice_1)}) == 2
        space.Module_InputChoice_1.hp_choice.assign([0])
        assert len(space.edges & {(space.Module_Input_1, space.Module_InputChoice_1),
                                  (space.Module_Input_2, space.Module_InputChoice_1)}) == 0
        assert len(space.edges & {(space.Module_Input_1, space.Module_Or_1)}) == 1
        assert len(space.edges & {(space.Module_Input_2, space.Module_Or_1)}) == 0

        space.Module_Or_1.hp_or.assign(0)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_Input_2', 'ID_test_sepconv5x5_relu_0_',
                       'ID_test_sepconv5x5_sepconv2d_0', 'ID_test_sepconv5x5_bn_0_',
                       'ID_test_sepconv5x5_relu_1_', 'ID_test_sepconv5x5_sepconv2d_1',
                       'ID_test_sepconv5x5_bn_1_']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_Or_1.hp_or.assign(1)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_Input_2', 'ID_test_sepconv3x3_relu_0_',
                       'ID_test_sepconv3x3_sepconv2d_0', 'ID_test_sepconv3x3_bn_0_', 'ID_test_sepconv3x3_relu_1_',
                       'ID_test_sepconv3x3_sepconv2d_1', 'ID_test_sepconv3x3_bn_1_']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_Or_1.hp_or.assign(2)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_Input_2', 'ID_test_avgpooling3x3_avgpooling3x3_',
                       'Module_AlignFilters_1']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_Or_1.hp_or.assign(3)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_Input_2', 'ID_test_maxpooling3x3_', 'Module_AlignFilters_2']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_Or_1.hp_or.assign(4)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_Input_2', 'ID_test_identity']

    def test_enas_op(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                filters = 64
                in1 = Input(shape=(28, 28, 1,))
                conv = conv_op('normal', 0, 0, 'L', [in1, in1], filters)
                space.set_inputs([in1, in1])
                space.set_outputs(conv)
                return space

        space = get_space()
        global ids
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_InputChoice_1', 'Module_Or_1']

        assert len(space.edges & {(space.Module_Input_1, space.Module_InputChoice_1)}) == 1
        space.Module_InputChoice_1.hp_choice.assign([0])
        assert len(space.edges & {(space.Module_Input_1, space.Module_InputChoice_1)}) == 0
        assert len(space.edges & {(space.Module_Input_1, space.Module_Or_1)}) == 1

        space.Module_Or_1.hp_or.assign(0)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'ID_normal_C0_N0_L_sepconv5x5_relu_0_',
                       'ID_normal_C0_N0_L_sepconv5x5_sepconv2d_0', 'ID_normal_C0_N0_L_sepconv5x5_bn_0_',
                       'ID_normal_C0_N0_L_sepconv5x5_relu_1_', 'ID_normal_C0_N0_L_sepconv5x5_sepconv2d_1',
                       'ID_normal_C0_N0_L_sepconv5x5_bn_1_']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_Or_1.hp_or.assign(1)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'ID_normal_C0_N0_L_sepconv3x3_relu_0_',
                       'ID_normal_C0_N0_L_sepconv3x3_sepconv2d_0', 'ID_normal_C0_N0_L_sepconv3x3_bn_0_',
                       'ID_normal_C0_N0_L_sepconv3x3_relu_1_', 'ID_normal_C0_N0_L_sepconv3x3_sepconv2d_1',
                       'ID_normal_C0_N0_L_sepconv3x3_bn_1_']
        #
        # model = space.keras_model()
        # plot_model(model)

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_Or_1.hp_or.assign(2)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'ID_normal_C0_N0_L_avgpooling3x3_avgpooling3x3_',
                       'Module_AlignFilters_1']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_Or_1.hp_or.assign(3)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'ID_normal_C0_N0_L_maxpooling3x3_', 'Module_AlignFilters_2']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_Or_1.hp_or.assign(4)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'ID_normal_C0_N0_L_identity']

    def test_enas_node(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                filters = 64
                in1 = Input(shape=(28, 28, 1,))
                conv_node('normal', 0, 0, [in1, in1], filters)
                space.set_inputs(in1)
                return space

        space = get_space()
        assert len(space.edges & {(space.Module_Input_1, space.Module_InputChoice_1),
                                  (space.Module_Input_1, space.Module_InputChoice_2)}) == 2
        global ids
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_InputChoice_1', 'Module_InputChoice_2',
                       'Module_Or_1', 'Module_Or_2', 'ID_normal_C0_N0_add_']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_Or_1.hp_or.assign(0)

        space.Module_InputChoice_2.hp_choice.assign([1])
        space.Module_Or_2.hp_or.assign(1)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'ID_normal_C0_N0_L_sepconv5x5_relu_0_',
                       'ID_normal_C0_N0_R_sepconv3x3_relu_0_', 'ID_normal_C0_N0_L_sepconv5x5_sepconv2d_0',
                       'ID_normal_C0_N0_R_sepconv3x3_sepconv2d_0', 'ID_normal_C0_N0_L_sepconv5x5_bn_0_',
                       'ID_normal_C0_N0_R_sepconv3x3_bn_0_', 'ID_normal_C0_N0_L_sepconv5x5_relu_1_',
                       'ID_normal_C0_N0_R_sepconv3x3_relu_1_', 'ID_normal_C0_N0_L_sepconv5x5_sepconv2d_1',
                       'ID_normal_C0_N0_R_sepconv3x3_sepconv2d_1', 'ID_normal_C0_N0_L_sepconv5x5_bn_1_',
                       'ID_normal_C0_N0_R_sepconv3x3_bn_1_', 'ID_normal_C0_N0_add_']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_Or_1.hp_or.assign(0)

        space.Module_InputChoice_2.hp_choice.assign([1])
        space.Module_Or_2.hp_or.assign(3)

        model = space.keras_model()
        plot_model(model, to_file='test_enas_node.png', show_shapes=True)

    def test_enas_cell(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                filters = 64
                in1 = Input(shape=(28, 28, 1,))
                conv_cell('normal', 0, [in1, in1], filters, 5)
                space.set_inputs(in1)
                return space

        space = get_space()
        space.random_sample()
        model = space.keras_model()
        plot_model(model, to_file='test_enas_cell.png', show_shapes=True)
