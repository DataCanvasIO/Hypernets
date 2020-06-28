# Overview

Hypernets introduced a more abstract search space representation, taking into account the requirements of hyperparameter optimization and neural architecture search(NAS), making Hypernets a general framework that can adapt to various automated machine learning needs.

The figure below shows conceptual model of Hypernets.

<p align="center">
<img src="https://github.com/DataCanvasIO/Hypernets/raw/master/docs/source/images/hypernets_conceptual_model.png" width="100%"/>
</p>

## Key Components

### HyperSpace
The space of all feasible solutions for a model is called **Search Space**. HyperSpace is an abstract representation of the search space composed of `Parameter Space`, `Connection Space`, and `Module Space`. The general form of HyperSpace is a DAG, so it can represent ML pipeline and neural network architecture very flexibly.

### Seacher
Search algorithms that looking for a optimal solution in `HyperSpace` and generating samples for `HyperModel`.

### HyperModel
High-level interface for users to perform model search and training, as long as the defined search space and training data are passed in to get the best model. HyperModel is an abstract class that needs to implement a dedicated HyperModel for different frameworks or domains. For example, `HyperKeras` is used to automatically search for neural networks built with keras, and `HyperML` is used to automatically optimize ML pipeline composed of sklearn, xgboost, and lightgbm....

### Estimator
A specific `HyperModle` needs to be paired with a dedicated `Estimator` to fit and evaluate the sample given by the `HyperModel`. This sample may be a set of hyperparameters, a network architecture, or a mixture of them.
