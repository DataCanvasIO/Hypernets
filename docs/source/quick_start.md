# Quick-Start

## Installation Guide

### Requirements
**Python 3**: Hypernets requires Python version 3.6 or 3.7. 

### Installation

```shell script
pip install hypernets
```

***Verify installation***:
```shell script
python -c "from examples import smoke_testing;"
```


## Getting started

In current version, we provide `HyperKeras` and `HyperDT` two `HyperModel`, which can be used for hyper-parameter tuning and NAS for Keras and DeepTables respectively.

DeepTables is a deep learning toolkit for tabular data, you can learn more here: [https://github.com/DataCanvasIO/DeepTables](https://github.com/DataCanvasIO/DeepTables)

Basically, to search the best model only needs 4 steps:
* Step 1. Define `Search Space`
* Step 2. Select a `Searcher`
* Step 3. Select a `HyperModel`
* Step 4. Search and get the best model


## Examples

### HyperKeras
```python
from hypernets.searchers.mcts_searcher import *
from hypernets.frameworks.keras.layers import *
from hypernets.frameworks.keras.hyper_keras import HyperKeras
from hypernets.core.callbacks import SummaryCallback

# Define Search Space
def get_space():
    space = HyperSpace()
    with space.as_default():
        in1 = Input(shape=(10,))
        dense1 = Dense(10, activation=Choice(['relu', 'tanh', None]), use_bias=Bool())(in1)
        bn1 = BatchNormalization()(dense1)
        dropout1 = Dropout(Choice([0.3, 0.4, 0.5]))(bn1)
        Dense(2, activation='softmax', use_bias=True)(dropout1)
    return space

# Select a Searcher
mcts = MCTSSearcher(get_space, max_node_space=4)

# Select a HyperModel
hk = HyperKeras(mcts, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                callbacks=[SummaryCallback()])

x = np.random.randint(0, 10000, size=(100, 10))
y = np.random.randint(0, 2, size=(100), dtype='int')

# Search and get the best model
hk.search(x, y, x, y, max_trails=10)
assert hk.best_model
```


### HyperDT (Hyperparameter Tuning & NAS for DeepTables)

* Install DeepTables
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

# Define Search Space
# `mini_dt_space` is a canned search space

# Select a Searcher
searcher = EvolutionSearcher(mini_dt_space, 200, 100, regularized=True, candidates_size=30,
                             optimize_direction=OptimizeDirection.Maximize)
# searcher = MCTSSearcher(mini_dt_space, max_node_space=0,optimize_direction=OptimizeDirection.Maximize)
# searcher = RandomSearcher(mini_dt_space, optimize_direction=OptimizeDirection.Maximize)

# Select a HyperModel
hdt = HyperDT(searcher, callbacks=[SummaryCallback(), FileLoggingCallback(searcher)], reward_metric='AUC',
              earlystopping_patience=1, )

df = dsutils.load_adult()
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
X = df_train
y = df_train.pop(14)
y_test = df_test.pop(14)

# Search and get the best model
hdt.search(df_train, y, df_test, y_test, max_trails=100, batch_size=256, epochs=10, verbose=1, )
assert hdt.best_model
best_trial = hdt.get_best_trail()
```

