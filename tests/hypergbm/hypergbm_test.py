# -*- coding:utf-8 -*-
"""

"""

from hypernets.core.search_space import *
from hypernets.core.ops import *
from hypernets.frameworks.ml.preprocessing import Pipeline, SimpleImputer, StandardScaler
import pytest

ids = []


def get_id(m):
    ids.append(m.id)
    return True


def tow_inputs():
    s1 = SimpleImputer()
    s2 = SimpleImputer()
    s3 = StandardScaler()([s1, s2])
    return s3


def tow_outputs():
    s1 = SimpleImputer()
    s2 = SimpleImputer()(s1)
    s3 = StandardScaler()(s1)
    return s2


class Test_HyperGBM:

    def test_pipeline(self):
        def get_space():
            space = HyperSpace()
            with space.as_default():
                Pipeline([SimpleImputer(), StandardScaler()])
            return space

        def get_space_2inputs():
            space = HyperSpace()
            with space.as_default():
                Pipeline([tow_inputs(), StandardScaler()])
            return space

        def get_space_2outputs():
            space = HyperSpace()
            with space.as_default():
                Pipeline([tow_outputs()])
            return space

        global ids
        space = get_space()
        space.random_sample()

        ids = []
        space.traverse(get_id)
        assert ids == ['ID_Module_Pipeline_1_input', 'Module_SimpleImputer_1', 'Module_StandardScaler_1', 'ID_Module_Pipeline_1_output']
        assert space.ID_Module_Pipeline_1_input.output_id =='ID_Module_Pipeline_1_output'
        assert space.ID_Module_Pipeline_1_output.input_id =='ID_Module_Pipeline_1_input'


        with pytest.raises(AssertionError) as e:
            space = get_space_2inputs()
            space.random_sample()
            ids = []
            space.traverse(get_id)

        with pytest.raises(AssertionError) as e:
            space = get_space_2outputs()
            space.random_sample()
            ids = []
            space.traverse(get_id)
