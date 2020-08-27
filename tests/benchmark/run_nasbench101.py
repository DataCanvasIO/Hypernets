# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from nasbench import api
from hypernets.benchmark.nas_bench_101 import NasBench101
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='Path to the file nasbench.tfrecord')
args = parser.parse_args()

nasbench = api.NASBench(args.input_file)
hyn_nasbench = NasBench101(7, ops=['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'])

while True:
    space = hyn_nasbench.get_space()
    space.random_sample()

    matrix, ops = hyn_nasbench.sample2spec(space)
    assert matrix.shape == (7, 7)
    assert len(ops) == 7

    model_spec = api.ModelSpec(matrix=matrix, ops=ops)
    if nasbench.is_valid(model_spec):
        break

# Query this model from dataset, returns a dictionary containing the metrics
# associated with this model.
print('Querying an Inception-like model.')
data = nasbench.query(model_spec)
print(data)
