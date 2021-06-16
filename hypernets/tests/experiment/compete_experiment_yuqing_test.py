import json

from hypernets.experiment import CompeteExperiment
from hypernets.experiment import Experiment
from hypernets.tabular import dask_ex as dex
from hypernets.tests.model.plain_model_test import create_plain_model
from hypernets.tests.tabular.dask_transofromer_test import setup_dask

from hypernets.tabular.datasets import dsutils
from tabular_toolbox.datasets import dsutils as dsu

import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd

from dask import dataframe as dd
from dask.distributed import LocalCluster, Client
from sklearn.preprocessing import LabelEncoder


# A test for multiclass task
def experiment_with_Bike_Sharing(init_kwargs, run_kwargs, row_count=3000, with_dask=False):
	hyper_model = create_plain_model(with_encoder=True)
	X = dsutils.load_Bike_Sharing()
	if row_count is not None:
		X = X.head(row_count)
	X['count'] = LabelEncoder().fit_transform(X['count'])
	y = X.pop('count')

	if with_dask:
		# setup_dask(None) #Unable to work for no reason
		cluster = LocalCluster(processes=False)
		client = Client(cluster)
		X = dex.dd.from_pandas(X, npartitions=1)
		y = dex.dd.from_pandas(y, npartitions=1)

	X_train, X_test, y_train, y_test = \
		dex.train_test_split(X, y, test_size=0.3, random_state=9527)
	X_train, X_eval, y_train, y_eval = \
		dex.train_test_split(X_train, y_train, test_size=0.3, random_state=9527)
	
	init_kwargs = {
        'X_eval': X_eval, 'y_eval': y_eval, 'X_test': X_test,
        **init_kwargs
    }
	
	compete_experiment = CompeteExperiment(hyper_model, X_train, y_train, **init_kwargs)
	base_experiment = Experiment(hyper_model, X_train, y_train, **init_kwargs)

	mydict_compete = compete_experiment.get_data_character()
	mydict_base = base_experiment.get_data_character()

	assert mydict_base
	assert mydict_compete
	assert mydict_base['experimentType'] == 'base'
	assert mydict_compete['experimentType'] == 'compete'
	assert mydict_base['target']['taskType'] == 'multiclass'
	assert mydict_base['target']['freq'] == None
	assert mydict_base['target']['unique']
	assert mydict_base['target']['mean'] == None
	assert mydict_base['target']['max'] == None
	assert mydict_base['target']['min'] == None
	assert mydict_base['target']['stdev'] == None
	assert mydict_base['target']['dataType']
	assert len(mydict_base['targetDistribution']) <= 10
	assert mydict_base['datasetShape']['X_train']
	assert mydict_base['datasetShape']['y_train']
	assert mydict_base['datasetShape']['X_eval']
	assert mydict_base['datasetShape']['y_eval']
	assert mydict_base['datasetShape']['X_test']
	assert mydict_compete['featureDistribution']


# A test for binary task
def experiment_with_load_blood(init_kwargs, run_kwargs, row_count=3000, with_dask=False):
	hyper_model = create_plain_model(with_encoder=True)
	X = dsu.load_blood()
	if row_count is not None:
		X = X.head(row_count)
	X['Class'] = LabelEncoder().fit_transform(X['Class'])
	y = X.pop('Class')

	if with_dask:
		# setup_dask(None) #Unable to work for no reason
		cluster = LocalCluster(processes=False)
		client = Client(cluster)
		X = dex.dd.from_pandas(X, npartitions=1)
		y = dex.dd.from_pandas(y, npartitions=1)
	
	X_train, X_test, y_train, y_test = \
	dex.train_test_split(X, y, test_size=0.3, random_state=9527)
	X_train, X_eval, y_train, y_eval = \
	dex.train_test_split(X_train, y_train, test_size=0.3, random_state=9527)

	init_kwargs = {
	'X_eval': X_eval, 'y_eval': y_eval, 'X_test': X_test,
	**init_kwargs
    }

	compete_experiment = CompeteExperiment(hyper_model, X_train, y_train, **init_kwargs)
	base_experiment = Experiment(hyper_model, X_train, y_train, **init_kwargs)

	mydict_compete = compete_experiment.get_data_character()
	mydict_base = base_experiment.get_data_character()

	assert mydict_base
	assert mydict_compete
	assert mydict_base['experimentType'] == 'base'
	assert mydict_compete['experimentType'] == 'compete'
	assert mydict_base['target']['taskType'] == 'binary'
	assert mydict_base['target']['freq'] != None
	assert mydict_base['target']['unique'] == 2
	assert mydict_base['target']['mean'] == None
	assert mydict_base['target']['max'] == None
	assert mydict_base['target']['min'] == None
	assert mydict_base['target']['stdev'] == None
	assert mydict_base['target']['dataType']
	assert len(mydict_base['targetDistribution']) <= 10
	assert mydict_base['datasetShape']['X_train']
	assert mydict_base['datasetShape']['y_train']
	assert mydict_base['datasetShape']['X_eval']
	assert mydict_base['datasetShape']['y_eval']
	assert mydict_base['datasetShape']['X_test']
	assert mydict_compete['featureDistribution']

# A test for regression task
def experiment_with_MPG_Data_Set(init_kwargs, run_kwargs, row_count=3000, with_dask=False):
	hyper_model = create_plain_model(with_encoder=True)
	datasetPath = tf.keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
	columnNames = ["MPG","Cylinders","Displacement","Horsepower","Weight","Accleration","Model Year","Origin"]
	rawDataset = pd.read_csv(datasetPath, names=columnNames, na_values="?",comment="\t",sep=" ",skipinitialspace=True)
	X = rawDataset.copy()
	if row_count is not None:
		X = X.head(row_count)
	X['MPG'] = LabelEncoder().fit_transform(X['MPG'])
	y = X.pop('MPG')
	y = y.astype('float64')
	
	if with_dask:
		# setup_dask(None) #Unable to work for no reason
		cluster = LocalCluster(processes=False)
		client = Client(cluster)
		X = dex.dd.from_pandas(X, npartitions=1)
		y = dex.dd.from_pandas(y, npartitions=1)
	
	X_train, X_test, y_train, y_test = \
	dex.train_test_split(X, y, test_size=0.3, random_state=9527)
	X_train, X_eval, y_train, y_eval = \
	dex.train_test_split(X_train, y_train, test_size=0.3, random_state=9527)

	init_kwargs = {
	'X_eval': X_eval, 'y_eval': y_eval, 'X_test': X_test,
	**init_kwargs
    }

	compete_experiment = CompeteExperiment(hyper_model, X_train, y_train, **init_kwargs)
	base_experiment = Experiment(hyper_model, X_train, y_train, **init_kwargs)

	mydict_compete = compete_experiment.get_data_character()
	mydict_base = base_experiment.get_data_character()

	assert mydict_base
	assert mydict_compete
	assert mydict_base['experimentType'] == 'base'
	assert mydict_compete['experimentType'] == 'compete'
	assert mydict_base['target']['taskType'] == 'regression'
	assert mydict_base['target']['freq'] == None
	assert mydict_base['target']['unique']
	assert mydict_base['target']['mean'] != None
	assert mydict_base['target']['max'] != None
	assert mydict_base['target']['min'] != None
	assert mydict_base['target']['stdev'] != None
	assert mydict_base['target']['dataType'] == 'float'
	assert len(mydict_base['targetDistribution']) <= 10
	assert mydict_base['datasetShape']['X_train']
	assert mydict_base['datasetShape']['y_train']
	assert mydict_base['datasetShape']['X_eval']
	assert mydict_base['datasetShape']['y_eval']
	assert mydict_base['datasetShape']['X_test']
	assert mydict_compete['featureDistribution']


def test_regression_with_MPG_Data_Set():
	experiment_with_MPG_Data_Set({}, {})
	experiment_with_MPG_Data_Set({}, {}, with_dask=True)


def test_multiclass_with_Bike_Sharing():
	experiment_with_Bike_Sharing({}, {})
	experiment_with_Bike_Sharing({}, {}, with_dask=True)

def test_binary_with_load_blood():
	experiment_with_load_blood({}, {})
	experiment_with_load_blood({}, {}, with_dask=True)

