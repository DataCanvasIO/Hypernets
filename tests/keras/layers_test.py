# -*- coding:utf-8 -*-
"""

"""

from hypernets.frameworks.keras.layers import *
from tensorflow.keras import models


class Test_KerasLayers():
    def test_input(self):
        space = HyperSpace()
        with space.as_default():
            input1 = Input(shape=(3000,))
            input2 = Input(shape=Choice([100, 200]), sparse=Bool())
            assert space.Param_Constant_1.alias == 'Module_Input_1.shape'
            assert space.Param_Choice_1.alias == 'Module_Input_2.shape'
            assert space.Param_Bool_1.alias == 'Module_Input_2.sparse'
            output = input1.compile_and_forward()
            assert output.shape, (None, 3000)

    def test_hyper_dense(self):
        space = HyperSpace()
        with space.as_default():
            input = Input(shape=(3000,))
            dense = Dense(units=Int(100, 200))
            dense(input)
            space.random_sample()
            assert dense.is_params_ready == True
            x = input.compile_and_forward()
            assert x.shape, (None, 3000)
            x = dense.compile_and_forward(x)
            assert x.shape, (None, dense.param_values['units'])

    def test_model_no_hp(self):
        space = HyperSpace()
        with space.as_default():
            in1 = Input(shape=(10,))
            in2 = Input(shape=(20,))
            in3 = Input(shape=(1,))
            concat = Concatenate()([in1, in2, in3])
            dense1 = Dense(10, activation='relu', use_bias=True)(concat)
            bn1 = BatchNormalization()(dense1)
            dropout1 = Dropout(0.3)(bn1)
            output = Dense(2, activation='relu', use_bias=True)(dropout1)

        model = space.keras_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        x1 = np.random.randint(0, 10000, size=(100, 10))
        x2 = np.random.randint(0, 100, size=(100, 20))
        x3 = np.random.normal(1.0, 100.0, size=(100))
        y = np.random.randint(0, 2, size=(100), dtype='int')
        x = [x1, x2, x3]
        history = model.fit(x=x, y=y)
        assert history

    def test_model_with_hp(self):
        space = HyperSpace()
        with space.as_default():
            in1 = Input(shape=(10,))
            in2 = Input(shape=(20,))
            in3 = Input(shape=(1,))
            concat = Concatenate()([in1, in2, in3])
            dense1 = Dense(10, activation=Choice(['relu', 'tanh', None]), use_bias=Bool())(concat)
            bn1 = BatchNormalization()(dense1)
            dropout1 = Dropout(Choice([0.3, 0.4, 0.5]))(bn1)
            output = Dense(2, activation=Choice(['relu', 'softmax']), use_bias=True)(dropout1)

        space.random_sample()
        print(space.params_summary())

        model = space.keras_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        x1 = np.random.randint(0, 10000, size=(100, 10))
        x2 = np.random.randint(0, 100, size=(100, 20))
        x3 = np.random.normal(1.0, 100.0, size=(100))
        y = np.random.randint(0, 2, size=(100), dtype='int')
        x = [x1, x2, x3]
        history = model.fit(x=x, y=y)
        assert history

    def tests_compile_space(self):
        space = HyperSpace()
        with space.as_default():
            input = Input(shape=(3000,))
            dense = Dense(units=Int(100, 200))
            dense(input)
            dense2 = Dense(units=2, name='dense_output')
            dense2(dense)

        space.random_sample()
        assert dense.is_params_ready == True
        compiled_space, _ = space.compile_and_forward()
        assert compiled_space.space_id == space.space_id

        outputs = compiled_space.get_outputs()
        assert len(outputs) == 1
        assert outputs[0].output.shape, (None, space.Module_Dense_2.param_values['units'])

        model = models.Model(inputs=compiled_space.Module_Input_1.output, outputs=outputs[0].output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        assert model.get_layer('Module_Input_1').output.shape[1] == 3000
        assert model.get_layer('Module_Dense_1').output.shape[1] == compiled_space.Module_Dense_1.param_values['units']
        assert model.get_layer('dense_output').output.shape[1] == 2
