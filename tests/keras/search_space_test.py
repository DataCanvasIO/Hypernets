# -*- coding:utf-8 -*-
"""

"""

from hypernets.frameworks.keras.dnn_search_space import dnn_block, dnn_search_space
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.callbacks import SummaryCallback
from hypernets.frameworks.keras.models import HyperKeras
import numpy as np


class Test_Dnn_Space():
    def test_dnn_space_hyper_model(self):
        rs = RandomSearcher(lambda: dnn_search_space(input_shape=10, output_units=2, output_activation='sigmoid'),
                            optimize_direction='max')
        hk = HyperKeras(rs, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                        callbacks=[SummaryCallback()])

        x = np.random.randint(0, 10000, size=(100, 10))
        y = np.random.randint(0, 2, size=(100), dtype='int')

        hk.search(x, y, x, y, max_trails=3)
        assert hk.best_model

    def test_dnn_space(self):
        space = dnn_search_space(input_shape=10, output_units=2, output_activation='sigmod')
        space.random_sample()
        ids = []
        assert space.combinations

        def get_id(m):
            ids.append(m.id)
            return True

        space.traverse(get_id)
        assert ids
