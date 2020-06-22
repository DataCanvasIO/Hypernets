# Hypernets


## Hypernets: An Open Source Automated Machine Learning Framework
HyperNets is a general AutoML framework that can meet various needs such as feature engineering, hyperparameter optimization, and neural architecture search, thereby helping users achieve the end-to-end automated machine learning pipeline.
 


## Overview
### Conceptual Model
<p align="center">
<img src="docs/images/hypernets_conceptual_model.png" width="100%"/>
</p>

### Illustration of the Search Space 
<p align="center">
<img src="docs/images/hypernets_search_space.png" width="100%"/>
</p>


## Installation
```shell script
pip install hypernets
```

***Verify installation***:
```shell script
python -c "from examples import smoke_testing;"
```


## Examples

### HyperKeras
```python
from hypernets.searchers.mcts_searcher import *
from hypernets.frameworks.keras.layers import *
from hypernets.frameworks.keras.models import HyperKeras
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

### HyperDT (Hyperparameter Tuning & NAS for DeepTables)

#### Install DeepTables
```shell script
pip install deeptables
```

```python
from contrib.deeptables.models import *
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.searcher import OptimizeDirection
from hypernets.core.callbacks import SummaryCallback, FileLoggingCallback
from hypernets.searchers.mcts_searcher import MCTSSearcher
from hypernets.searchers.evolution_searcher import EvolutionSearcher
from hypernets.core.trial import DiskTrailStore
from deeptables.datasets import dsutils
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