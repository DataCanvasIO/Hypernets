HyperModel
=============

HyperModel is the core element of the Hypernets architecture. It is a high-level interface which can access the defined search space and the training data to perform model search and model training. The figure below shows the HyperModel search sequence. For every loop, the `HyperModel` explores hyperparameter samples from `Searcher`, calls the `Estimator` to perform fitting and evaluation operations, then updates the metric score to `Searcher` for further optimization.

.. image:: images/hyper_model_search_sequence.png
   :width: 600
   :align: center
   :alt: search sequence


Customize HyperModel
-------------------------

To customize HyperModel, two components are required:

* HyperModel: subclass of *hypernets.model.HyperModel*. It creates a new estimator instance with the defined hyperparameter samples. Meanwhile, it loads the trained estimator from storage.

* Estimator: subclass of  *hypernets.model.Estimator*. It contains subfunctions: model fitting, evaluation, prediction, etc.

As a start point, refer to the examples  *hypernets.examples.plain_model.PlainModel* and *hypernets.examples.plain_model.PlainEstimator*.

For more details, see `DeepTables <https://github.com/DataCanvasIO/DeepTables>`_, `HyperGBM <https://github.com/DataCanvasIO/HyperGBM>`_, `HyperKeras <https://github.com/DataCanvasIO/HyperKeras>`_.
