import json

from hypernets.experiment import CompeteExperiment
from hypernets.experiment import Experiment
from hypernets.tests.model.plain_model_test import create_plain_model
from deeptables.datasets import dsutils

import numpy as np
import pandas as pd

from dask import dataframe as dd
from dask.distributed import LocalCluster, Client
from hypergbm import make_experiment
from tabular_toolbox.datasets import dsutils as dsu

def experiment_pandas():
	# # Discontinuous data
	# df_train = dsutils.load_adult()
	# y = df_train.pop(14)
	# X = df_train

	# Continuous data
	data = list(range(0,301))
	data = data + list(range(293, 298))
	y = pd.Series(data)
	X = dsutils.load_adult()

	hyper_model = create_plain_model(with_encoder=True)
	X_train = X
	y_train = y
	compete_experiment = CompeteExperiment(hyper_model, X_train, y_train)
	base_experiment = Experiment(hyper_model, X_train, y_train)
	mydict_compete = json.dumps(compete_experiment.get_data_character())
	mydict_base = json.dumps(base_experiment.get_data_character())
	print(mydict_compete)
	print(mydict_base)

def experiment_dask():
	cluster = LocalCluster(processes=False)
	client = Client(cluster)

	hyper_model = create_plain_model(with_encoder=True)

	# # y_train's dtype is in64, but it shows the result of classification
	# train_data = dd.from_pandas(dsu.load_blood(), npartitions=1)
	# y_train = train_data.pop('Class')
	# X_train = train_data
	

	# y_train's dtype is int or float
	# train_data = dd.from_pandas(dsu.load_blood(), npartitions=1)
	# X_train = train_data
	# data = list(range(0,301))
	# data = data + list(range(293, 298))
	# y_train_pandas = pd.Series(data)
	# y_train = dd.from_pandas(y_train_pandas, npartitions=1)

	# y_train's dtype is object
	df_train = dd.from_pandas(dsutils.load_adult(), npartitions=1)
	y_train = df_train.pop(14)
	X_train = df_train

	compete_experiment = CompeteExperiment(hyper_model, X_train, y_train)
	base_experiment = Experiment(hyper_model, X_train, y_train)
	mydict_compete = compete_experiment.get_data_character()
	mydict_base = base_experiment.get_data_character()
	print(mydict_compete)
	print(mydict_base)

experiment_pandas()
experiment_dask()
