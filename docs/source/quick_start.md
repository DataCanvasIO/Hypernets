# Quick-Start

## Installation

Python version 3.6 or above is necessary before installing Hypernets. 

### Conda

Install Hypernets with `conda` from the channel *conda-forge*:

```bash
conda install -c conda-forge hypernets
```

### Pip

Install Hypernets with `pip` command:

```bash
pip install hypernets
```

Optional, to run Hypernets in JupyterLab notebooks, install Hypernets and JupyterLab with command:
```bash
pip install hypernets[notebook]
```
See more about [Hypernets jupyter notebook widget](jupyter_widget.md)

Optional, to run Hypernets in distributed Dask cluster, install Hypernets with command:
```bash
pip install hypernets[dask]
```

Optional, to support simplified Chinese in feature generation, install `jieba` package before run Hypernets, or install Hypernets with command:
```bash
pip install hypernets[zhcn]
```

Optional, install all Hypernets components and dependencies with one command:
```bash
pip install hypernets[all]
```
``

***Verify installation***:
```bash
python -m hypernets.examples.smoke_testing
```


## Getting started

In current version, we provide `PlainModel` (a plain `HyperModel` implementation), which can be used  for hyper-parameter tuning with sklearn machine learning algorithms.

Basically, to search the best model only needs 4 steps:
* Step 1. Define `Search Space`
* Step 2. Select a `Searcher`
* Step 3. Select a `HyperModel`
* Step 4. Search and get the best model


### Define Search space
Firstly, we define a search space for hyper-parameters of *DecisionTreeClassifier* and *MLPClassifier*:

```python
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from hypernets.core import get_random_state
from hypernets.core.ops import ModuleChoice, HyperInput, ModuleSpace
from hypernets.core.search_space import HyperSpace, Choice, Int


def my_search_space(enable_dt=True, enable_mlp=True):
    space = HyperSpace()

    with space.as_default():
        hyper_input = HyperInput(name='input1')

        estimators = []
        if enable_dt:
            estimators.append(dict(
                cls=DecisionTreeClassifier,
                criterion=Choice(["gini", "entropy"]),
                splitter=Choice(["best", "random"]),
                max_depth=Choice([None, 3, 5, 10, 20, 50]),
                random_state=get_random_state(),
            ))

        if enable_mlp:
            estimators.append(dict(
                cls=MLPClassifier,
                max_iter=Int(500, 5000, step=500),
                activation=Choice(['identity', 'logistic', 'tanh', 'relu']),
                solver=Choice(['lbfgs', 'sgd', 'adam']),
                learning_rate=Choice(['constant', 'invscaling', 'adaptive']),
                random_state=get_random_state(),
            ))

        modules = [ModuleSpace(name=f'{e["cls"].__name__}', **e) for e in estimators]
        outputs = ModuleChoice(modules)(hyper_input)
        space.set_inputs(hyper_input)

    return space
```


### Training with PlainModel

Turning and scoring with `my_search_space` for `heart_disease_uci` dataset.

```python

def train():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from hypernets.core.callbacks import SummaryCallback
    from hypernets.examples.plain_model import PlainModel
    from hypernets.searchers import make_searcher
    from hypernets.tabular.datasets import dsutils

    X = dsutils.load_heart_disease_uci()
    y = X.pop('target')

    X_train, X_eval, y_train, y_eval = \
        train_test_split(X, y, test_size=0.3)
    
    # make MCTS searcher
    searcher = make_searcher('mcts', my_search_space, optimize_direction='max')
    callbacks = [SummaryCallback()]
    
    # create HyperModel and do 'search' action
    hm = PlainModel(searcher=searcher, reward_metric='f1', callbacks=callbacks)
    hm.search(X_train, y_train, X_eval, y_eval, )
    
    # get best estimator
    best = hm.get_best_trial()
    estimator = hm.final_train(best.space_sample, X_train, y_train)
    
    # scoring
    y_pred = estimator.predict(X_eval)
    print(classification_report(y_eval, y_pred))


if __name__ == '__main__':
    train()

```

Run the example, we will get console output:
```console
17:36:56 I hypernets.u.common.py 147 - 2 class detected, {0, 1}, so inferred as a [binary classification] task
17:36:56 I hypernets.c.meta_learner.py 22 - Initialize Meta Learner: dataset_id:e10ae1d61123d55062f7d3b64b79a6e1
17:36:56 I hypernets.c.callbacks.py 235 - 
Trial No:1
--------------------------------------------------------------
(0) Module_ModuleChoice_1.hp_or:                            0
(1) DecisionTreeClassifier.criterion:                    gini
(2) DecisionTreeClassifier.splitter:                     best
(3) DecisionTreeClassifier.max_depth:                       5
--------------------------------------------------------------
...

              precision    recall  f1-score   support

           0       0.68      0.67      0.67        42
           1       0.72      0.73      0.73        49

    accuracy                           0.70        91
   macro avg       0.70      0.70      0.70        91
weighted avg       0.70      0.70      0.70        91
```
