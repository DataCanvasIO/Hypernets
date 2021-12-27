# Overview

Hypernets is a general automated machine learning framework, based on which it can implement automatic optimization tools for various machine learning frameworks and libraries, including deep learning frameworks such as tensorflow, keras, pytorch, and machine learning libraries like sklearn, lightgbm, xgboost, etc.
We introduced an abstract search space representation, taking into account the requirements of hyperparameter optimization and neural architecture search(NAS), making Hypernets a general framework that can adapt to various automated machine learning needs.

The figure below shows conceptual model of Hypernets.

![hypernets_conceptual_model](images/hypernets_conceptual_model.png)

## Key Components

### HyperSpace
The space of all feasible solutions for a model is called **Search Space**. HyperSpace is an abstract representation of the search space composed of `Parameter Space`, `Connection Space`, and `Module Space`. The general form of HyperSpace is a Directed Acyclic Graph (DAG), so it can represent flexibly the ML pipeline and the neural network architecture.

### Searcher
Various search strategies are involved in the `Searcher`, which iteratively looks for hyperparameters in `HyperSpace` and transfers to `HyperModel`. Combined with `Estimator`, it could find the optimal model.

### HyperModel
`HyperModel` is a high-level interface which can access the defined search space and the training data to perform model search and model training. HyperModel is formed as an abstract class that can be implemented for different frameworks and domains. The well developed HyperModels are `HyperGBM` and `HyperDT`, which can process tabular dataset by various GBM algorithms (lightGBM, XGBoost, CatBoost) and self-built algorithm DeepTable respectively. `HyperKeras`, as another HyperModel, could automatically search for neural networks. 

### Estimator
Every specific HyperModel is paired with a dedicated `Estimator`. The Estimator receives either a set of hyperparametres, or a network architecture from HyperModel and sends back rewards (metric scores) after fitting and evaluation.

### Experiment
`Experiment` is the playground where prepares the training and testing data, and obtains the optimized estimator through HyperModel.
