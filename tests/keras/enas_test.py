# -*- coding:utf-8 -*-
"""

"""
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from hypernets.core.ops import *
from hypernets.core.search_space import HyperSpace
from hypernets.frameworks.keras.enas_common_ops import sepconv3x3, sepconv5x5, avgpooling3x3, \
    maxpooling3x3, identity, conv_cell, conv_node, conv_layer
from hypernets.frameworks.keras.enas_micro import enas_micro_search_space
from hypernets.frameworks.keras.layers import Input
from tests import test_output_dir
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
                or1 = ModuleChoice([sepconv5x5(name_prefix, filters),
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
        assert ids == ['Module_Input_1', 'Module_Input_2', 'Module_InputChoice_1', 'Module_ModuleChoice_1']

        assert len(space.edges & {(space.Module_Input_1, space.Module_InputChoice_1),
                                  (space.Module_Input_2, space.Module_InputChoice_1)}) == 2
        space.Module_InputChoice_1.hp_choice.assign([0])
        assert len(space.edges & {(space.Module_Input_1, space.Module_InputChoice_1),
                                  (space.Module_Input_2, space.Module_InputChoice_1)}) == 0
        assert len(space.edges & {(space.Module_Input_1, space.Module_ModuleChoice_1)}) == 1
        assert len(space.edges & {(space.Module_Input_2, space.Module_ModuleChoice_1)}) == 0

        space.Module_ModuleChoice_1.hp_or.assign(0)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_Input_2', 'ID_test_sepconv5x5_relu_0_',
                       'ID_test_sepconv5x5_sepconv2d_0', 'ID_test_sepconv5x5_bn_0_',
                       'ID_test_sepconv5x5_relu_1_', 'ID_test_sepconv5x5_sepconv2d_1',
                       'ID_test_sepconv5x5_bn_1_']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_ModuleChoice_1.hp_or.assign(1)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_Input_2', 'ID_test_sepconv3x3_relu_0_',
                       'ID_test_sepconv3x3_sepconv2d_0', 'ID_test_sepconv3x3_bn_0_', 'ID_test_sepconv3x3_relu_1_',
                       'ID_test_sepconv3x3_sepconv2d_1', 'ID_test_sepconv3x3_bn_1_']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_ModuleChoice_1.hp_or.assign(2)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_Input_2', 'ID_test_avgpooling3x3_pool_']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_ModuleChoice_1.hp_or.assign(3)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_Input_2', 'ID_test_maxpooling3x3_pool_']

        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_ModuleChoice_1.hp_or.assign(4)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_Input_2', 'ID_test_identity']

    def test_enas_op(self):
        hp_dict = {}

        def get_space():
            space = HyperSpace()
            with space.as_default():
                filters = 64
                in1 = Input(shape=(28, 28, 1,))
                conv = conv_cell(hp_dict, 'normal', 0, 0, 'L', [in1, in1], filters)
                space.set_inputs([in1, in1])
                space.set_outputs(conv)
                return space

        space = get_space()
        assert len(hp_dict.items()) == 2
        global ids
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_InputChoice_1', 'Module_ModuleChoice_1']

        assert len(space.edges & {(space.Module_Input_1, space.Module_InputChoice_1)}) == 1
        space.Module_InputChoice_1.hp_choice.assign([0])
        assert len(space.edges & {(space.Module_Input_1, space.Module_InputChoice_1)}) == 0
        assert len(space.edges & {(space.Module_Input_1, space.Module_ModuleChoice_1)}) == 1

        space.Module_ModuleChoice_1.hp_or.assign(0)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'ID_normal_C0_N0_L_sepconv5x5_relu_0_',
                       'ID_normal_C0_N0_L_sepconv5x5_sepconv2d_0', 'ID_normal_C0_N0_L_sepconv5x5_bn_0_',
                       'ID_normal_C0_N0_L_sepconv5x5_relu_1_', 'ID_normal_C0_N0_L_sepconv5x5_sepconv2d_1',
                       'ID_normal_C0_N0_L_sepconv5x5_bn_1_']
        hp_dict = {}
        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_ModuleChoice_1.hp_or.assign(1)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'ID_normal_C0_N0_L_sepconv3x3_relu_0_',
                       'ID_normal_C0_N0_L_sepconv3x3_sepconv2d_0', 'ID_normal_C0_N0_L_sepconv3x3_bn_0_',
                       'ID_normal_C0_N0_L_sepconv3x3_relu_1_', 'ID_normal_C0_N0_L_sepconv3x3_sepconv2d_1',
                       'ID_normal_C0_N0_L_sepconv3x3_bn_1_']
        #
        # model = space.keras_model()
        # plot_model(model)
        hp_dict = {}
        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_ModuleChoice_1.hp_or.assign(2)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'ID_normal_C0_N0_L_avgpooling3x3_pool_']
        hp_dict = {}
        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_ModuleChoice_1.hp_or.assign(3)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'ID_normal_C0_N0_L_maxpooling3x3_pool_']

        hp_dict = {}
        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_ModuleChoice_1.hp_or.assign(4)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'ID_normal_C0_N0_L_identity']

    def test_enas_node(self):
        hp_dict = {}

        def get_space():
            space = HyperSpace()
            with space.as_default():
                filters = 64
                in1 = Input(shape=(28, 28, 1,), dtype='float32')
                conv_node(hp_dict, 'normal', 0, 0, [in1, in1], filters)
                space.set_inputs(in1)
                return space

        space = get_space()
        assert len(hp_dict.items()) == 4
        assert len(space.edges & {(space.Module_Input_1, space.Module_InputChoice_1),
                                  (space.Module_Input_1, space.Module_InputChoice_2)}) == 2
        global ids
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'Module_InputChoice_1', 'Module_InputChoice_2',
                       'Module_ModuleChoice_1', 'Module_ModuleChoice_2', 'ID_normal_C0_N0_add_']
        hp_dict = {}
        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_ModuleChoice_1.hp_or.assign(0)

        space.Module_InputChoice_2.hp_choice.assign([1])
        space.Module_ModuleChoice_2.hp_or.assign(1)
        ids = []
        space.traverse(get_id)
        assert ids == ['Module_Input_1', 'ID_normal_C0_N0_L_sepconv5x5_relu_0_',
                       'ID_normal_C0_N0_R_sepconv3x3_relu_0_', 'ID_normal_C0_N0_L_sepconv5x5_sepconv2d_0',
                       'ID_normal_C0_N0_R_sepconv3x3_sepconv2d_0', 'ID_normal_C0_N0_L_sepconv5x5_bn_0_',
                       'ID_normal_C0_N0_R_sepconv3x3_bn_0_', 'ID_normal_C0_N0_L_sepconv5x5_relu_1_',
                       'ID_normal_C0_N0_R_sepconv3x3_relu_1_', 'ID_normal_C0_N0_L_sepconv5x5_sepconv2d_1',
                       'ID_normal_C0_N0_R_sepconv3x3_sepconv2d_1', 'ID_normal_C0_N0_L_sepconv5x5_bn_1_',
                       'ID_normal_C0_N0_R_sepconv3x3_bn_1_', 'ID_normal_C0_N0_add_']
        hp_dict = {}
        space = get_space()
        space.Module_InputChoice_1.hp_choice.assign([0])
        space.Module_ModuleChoice_1.hp_or.assign(0)

        space.Module_InputChoice_2.hp_choice.assign([1])
        space.Module_ModuleChoice_2.hp_or.assign(3)

        model = space.keras_model()
        plot_model(model, to_file=f'{test_output_dir}/test_enas_node.png', show_shapes=True)

    def test_enas_layer(self):
        hp_dict = {}

        def get_space():
            space = HyperSpace()
            with space.as_default():
                filters = 64
                in1 = Input(shape=(28, 28, 1,))
                conv_layer(hp_dict, 'normal', 0, [in1, in1], filters, 5)
                space.set_inputs(in1)
                return space

        space = get_space()
        assert len(hp_dict.items()) == 20
        space.random_sample()
        model = space.keras_model()
        plot_model(model, to_file=f'{test_output_dir}/test_enas_cell.png', show_shapes=True)

    def test_enas_micro(self):
        hp_dict = {}
        space = enas_micro_search_space(arch='NNRNNR', hp_dict=hp_dict)
        assert len(hp_dict.items()) == 32
        assert space.combinations

        space.random_sample()
        model = space.keras_model()
        plot_model(model, to_file=f'{test_output_dir}/test_enas_micro.png', show_shapes=True)
