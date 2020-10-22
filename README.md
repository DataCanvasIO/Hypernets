# Hypernets

[![Python Versions](https://img.shields.io/pypi/pyversions/hypernets.svg)](https://pypi.org/project/hypernets)
[![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-2.0+-blue.svg)](https://pypi.org/project/hypernets)
[![Downloads](https://pepy.tech/badge/hypernets)](https://pepy.tech/project/hypernets)
[![PyPI Version](https://img.shields.io/pypi/v/hypernets.svg)](https://pypi.org/project/hypernets)

## Hypernets: A General Automated Machine Learning Framework
Hypernets is a general AutoML framework, based on which it can implement automatic optimization tools for various machine learning frameworks and libraries, including deep learning frameworks such as tensorflow, keras, pytorch, and machine learning libraries like sklearn, lightgbm, xgboost, etc.
We introduced an abstract search space representation, taking into account the requirements of hyperparameter optimization and neural architecture search(NAS), making Hypernets a general framework that can adapt to various automated machine learning needs.


## Overview
### Conceptual Model
<p align="center">
<img src="docs/source/images/hypernets_conceptual_model.png" width="100%"/>
</p>

### Illustration of the Search Space 
<p align="center">
<img src="docs/source/images/hypernets_search_space.png" width="100%"/>
</p>


## Installation
```shell script
pip install hypernets
```

***Verify installation***:
```shell script
python -c "from examples import smoke_testing;"
```

## Hypernets related projects

* [HyperGBM](https://github.com/DataCanvasIO/HyperGBM): A full pipeline AutoML tool integrated various GBM models.
* [HyperDT/DeepTables](https://github.com/DataCanvasIO/DeepTables): An AutoDL tool for tabular data.
* [HyperKeras](https://github.com/DataCanvasIO/HyperKeras): An AutoDL tool for Neural Architecture Search and Hyperparameter Optimization on Tensorflow and Keras.
* [Cooka](https://github.com/DataCanvasIO/Cooka): Lightweight interactive AutoML system.
* [Hypernets](https://github.com/DataCanvasIO/Hypernets): A general automated machine learning framework.

![DataCanvas AutoML Toolkit](docs/source/images/datacanvas_automl_toolkit.png)

## Neural Architecture Search
* [Define A DNN Search Space](https://hypernets.readthedocs.io/en/latest/nas.html#define-a-dnn-search-space)
* [Define A CNN Search Space](https://hypernets.readthedocs.io/en/latest/nas.html#define-a-cnn-search-space)
* [Define An ENAS Micro Search Space](https://hypernets.readthedocs.io/en/latest/nas.html#define-an-enas-micro-search-space)

## Examples

### HyperKeras
```python
from hypernets.searchers.mcts_searcher import *
from hypernets.frameworks.keras.layers import *
from hypernets.frameworks.keras.hyper_keras import HyperKeras
from hypernets.core.callbacks import SummaryCallback
def get_space():
    space = HyperSpace()
    with space.as_default():
        in1 = Input(shape=(10,))
        dense1 = Dense(10, activation=Choice(['relu', 'tanh', None]), use_bias=Bool())(in1)
        bn1 = BatchNormalization()(dense1)
        dropout1 = Dropout(Choice([0.3, 0.4, 0.5]))(bn1)
        Dense(2, activation='softmax', use_bias=True)(dropout1)
    return space

mcts = MCTSSearcher(get_space, max_node_space=4)
hk = HyperKeras(mcts, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                callbacks=[SummaryCallback()])

x = np.random.randint(0, 10000, size=(100, 10))
y = np.random.randint(0, 2, size=(100), dtype='int')

hk.search(x, y, x, y, max_trails=10)
assert hk.best_model

```

### HyperGBM
```python
from hypernets.frameworks.ml.hyper_gbm import HyperGBM
from hypernets.frameworks.ml.common_ops import get_space_num_cat_pipeline_complex
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.callbacks import *
from hypernets.core.searcher import OptimizeDirection
from hypernets.frameworks.ml.datasets import dsutils
from sklearn.model_selection import train_test_split

rs = RandomSearcher(get_space_num_cat_pipeline_complex, optimize_direction=OptimizeDirection.Maximize)
hk = HyperGBM(rs, task='classification', reward_metric='accuracy',
              callbacks=[SummaryCallback(), FileLoggingCallback(rs)])

df = dsutils.load_bank()
df.drop(['id'], axis=1, inplace=True)
X_train, X_test = train_test_split(df.head(1000), test_size=0.2, random_state=42)
y_train = X_train.pop('y')
y_test = X_test.pop('y')

hk.search(X_train, y_train, X_test, y_test, max_trails=30)
assert hk.best_model
best_trial = hk.get_best_trail()

estimator = hk.final_train(best_trial.space_sample, X_train, y_train)
score = estimator.predict(X_test)
result = estimator.evaluate(X_test, y_test)
```

### HyperDT (Hyperparameter Tuning & NAS for DeepTables)

#### Install DeepTables
```shell script
pip install deeptables
```

```python
from deeptables.models.hyper_dt import mini_dt_space, HyperDT
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.searcher import OptimizeDirection
from hypernets.core.callbacks import SummaryCallback, FileLoggingCallback
from hypernets.searchers.mcts_searcher import MCTSSearcher
from hypernets.searchers.evolution_searcher import EvolutionSearcher
from hypernets.core.trial import DiskTrailStore
from hypernets.frameworks.ml.datasets import dsutils
from sklearn.model_selection import train_test_split

disk_trail_store = DiskTrailStore('~/trail_store')

# searcher = MCTSSearcher(mini_dt_space, max_node_space=0,optimize_direction=OptimizeDirection.Maximize)
# searcher = RandomSearcher(mini_dt_space, optimize_direction=OptimizeDirection.Maximize)
searcher = EvolutionSearcher(mini_dt_space, 200, 100, regularized=True, candidates_size=30,
                             optimize_direction=OptimizeDirection.Maximize)

hdt = HyperDT(searcher, callbacks=[SummaryCallback(), FileLoggingCallback(searcher)], reward_metric='AUC',
              earlystopping_patience=1, )

df = dsutils.load_adult()
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
X = df_train
y = df_train.pop(14)
y_test = df_test.pop(14)

hdt.search(df_train, y, df_test, y_test, max_trails=100, batch_size=256, epochs=10, verbose=1, )
assert hdt.best_model
best_trial = hdt.get_best_trail()
```


## DataCanvas
Hypernets is an open source project created by [DataCanvas](https://www.datacanvas.com/). 