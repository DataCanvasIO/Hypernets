# -*- coding:utf-8 -*-
"""

"""

from hypernets.frameworks.keras.layers import *
from hypernets.frameworks.keras.hyper_keras import HyperKeras
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection


class Test_HyperKeras():
    def get_space(self):
        space = HyperSpace()
        with space.as_default():
            in1 = Input(shape=(10,))
            in2 = Input(shape=(20,))
            in3 = Input(shape=(1,))
            concat = Concatenate()([in1, in2, in3])
            dense1 = Dense(10, activation=Choice(['relu', 'tanh', None]), use_bias=Bool())(concat)
            bn1 = BatchNormalization()(dense1)
            dropout1 = Dropout(Choice([0.3, 0.4, 0.5]))(bn1)
            output = Dense(2, activation='softmax', use_bias=True)(dropout1)
        return space

    def get_space_simple(self):
        space = HyperSpace()
        with space.as_default():
            in1 = Input(shape=(10,))
            dense1 = Dense(10, activation=Choice(['relu', 'tanh', None]), use_bias=Bool())(in1)
            bn1 = BatchNormalization()(dense1)
            dropout1 = Dropout(Choice([0.3, 0.4, 0.5]))(bn1)
            output = Dense(2, activation='softmax', use_bias=True)(dropout1)
        return space

    def get_x_y(self):
        x1 = np.random.randint(0, 10000, size=(100, 10))
        x2 = np.random.randint(0, 100, size=(100, 20))
        x3 = np.random.normal(1.0, 100.0, size=(100))
        y = np.random.randint(0, 2, size=(100), dtype='int')
        x = [x1, x2, x3]
        return x, y

    def get_x_y_1(self):
        x = np.random.randint(0, 10000, size=(100, 10))
        y = np.random.randint(0, 2, size=(100), dtype='int')
        return x, y

    def test_model_with_hp(self):
        rs = RandomSearcher(self.get_space, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperKeras(rs, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                        callbacks=[SummaryCallback()])

        x, y = self.get_x_y()
        hk.search(x, y, x, y, max_trails=3)
        assert hk.best_model
        best_trial = hk.get_best_trail()

        estimator = hk.final_train(best_trial.space_sample, x, y)
        score = estimator.predict(x)
        result = estimator.evaluate(x, y)
        assert len(score) == 100
        assert result

    def test_build_dataset_iter(self):
        rs = RandomSearcher(self.get_space, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperKeras(rs, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                        callbacks=[SummaryCallback()])
        x, y = self.get_x_y_1()

        ds_iter = hk.build_dataset_iter(x, y, batch_size=10)

        batch_counter = 0
        for x_b, y_b in ds_iter:
            # x_b, y_b = next()
            assert len(x_b) == 10
            assert len(y_b) == 10
            batch_counter += 1
        assert batch_counter == 10

        ds_iter = hk.build_dataset_iter(x, y, batch_size=32)

        batch_counter = 0
        for x_b, y_b in ds_iter:
            # x_b, y_b = next()
            if batch_counter < 3:
                assert len(x_b) == 32
                assert len(y_b) == 32
            else:
                assert len(x_b) == 4
                assert len(y_b) == 4
            batch_counter += 1
        assert batch_counter == 4

        ds_iter = hk.build_dataset_iter(x, y, batch_size=32, repeat_count=2)

        batch_counter = 0
        for x_b, y_b in ds_iter:
            # x_b, y_b = next()
            if batch_counter < 6:
                assert len(x_b) == 32
                assert len(y_b) == 32
            else:
                assert len(x_b) == 8
                assert len(y_b) == 8
            batch_counter += 1
        assert batch_counter == 7

    def test_fit_one_shot_model_epoch(self):
        rs = RandomSearcher(self.get_space_simple, optimize_direction=OptimizeDirection.Maximize)
        hk = HyperKeras(rs,
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'],
                        callbacks=[SummaryCallback()],
                        one_shot_mode=True,
                        one_shot_train_sampler=rs)
        x, y = self.get_x_y_1()
        hk.fit_one_shot_model_epoch(x, y)
