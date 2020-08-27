# NASBench: A Neural Architecture Search Dataset and Benchmark

## Setup

1.  Clone this repo.

```
# git clone https://github.com/google-research/nasbench
# clone from a fork to compatible with tensorflow 2.x
git clone https://github.com/jackguagua/nasbench
cd nasbench
```

2. Install the project along with dependencies.

```
pip install -e .
```

**Note:** the only required dependency is TensorFlow. The above instructions
will install the CPU version of TensorFlow to the virtualenv. For other install
options, see https://www.tensorflow.org/install/.

## Download the dataset

The full dataset (which includes all 5M data points at all 4 epoch lengths):

https://storage.googleapis.com/nasbench/nasbench_full.tfrecord

Size: ~1.95 GB, SHA256: `3d64db8180fb1b0207212f9032205064312b6907a3bbc81eabea10db2f5c7e9c`

---

Subset of the dataset with only models trained at 108 epochs:

https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord

Size: ~499 MB, SHA256: `4c39c3936e36a85269881d659e44e61a245babcb72cb374eacacf75d0e5f4fd1`


## Using the dataset

Example usage (see `example.py` for a full runnable example):

```python
# Load the data from file (this will take some time)
from nasbench import api

nasbench = api.NASBench('/path/to/nasbench.tfrecord')

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'


# Create an Inception-like module (5x5 convolution replaced with two 3x3
# convolutions).
model_spec = api.ModelSpec(
    # Adjacency matrix of the module
    matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
            [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
            [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
            [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
            [0, 0, 0, 0, 0, 0, 0]],   # output layer
    # Operations at the vertices of the module, matches order of matrix
    ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

# Query this model from dataset, returns a dictionary containing the metrics
# associated with this model.
data = nasbench.query(model_spec)
```

## Disclaimer

This is not an official Google product.
