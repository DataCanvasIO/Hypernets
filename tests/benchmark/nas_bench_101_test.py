# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from hypernets.benchmark.nas_bench_101 import NasBench101


# class Test_NASBench101():

    # def test_get_space(self):
    #     nasbench = NasBench101(7, ops=['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'])
    #     space = nasbench.get_space()
    #     space.random_sample()
    #     assert space.vectors
    #
    #     matrix, ops = nasbench.sample2spec(space)
    #     assert matrix.shape == (7, 7)
    #     assert len(ops) == 7
