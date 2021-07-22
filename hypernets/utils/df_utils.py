import json

import numpy as np
import pandas as pd
from dask import dataframe as dd

from hypernets.tabular import column_selector as col_se


def get_data_character(hyper_model, X_train, y_train, X_eval=None, y_eval=None, X_test=None, task=None):

	dtype2usagetype = {'object':'str', 'int64':'int', 'float64':'float', 'datetime64[ns]':'date', 'timedelta64[ns]':'date'}
	
	task, _ = hyper_model.infer_task_type(y_train) #This line is just used to test

	if isinstance(y_train, pd.core.series.Series):
		datatype_y = dtype2usagetype[str(y_train.dtypes)]

		Missing_y = y_train.isnull().sum().tolist()
		Unique_y = len(y_train.unique())

		if task == 'binary':
			Freq_y = y_train.value_counts()[0].tolist()
		else:
			Freq_y = None
		
		if task == 'regression':
			max_y = max(y_train)
			min_y = min(y_train)
			mean_y = pd.Series.mean(y_train)
			Stdev_y = y_train.std()
		else:
			max_y = None
			min_y = None
			mean_y = None
			Stdev_y = None

		if task == 'regression':
			Freq_y = None
			disc_y_num = None
			cont_y_num = 10
			interval = (max_y-min_y)/cont_y_num
			intervals = np.linspace(min_y, max_y, cont_y_num+1)
			Freqs_y = pd.cut(y_train, intervals).value_counts(sort=False)  # [0:cont_y_num]
			count = list(Freqs_y)
			region = list(map(lambda x: [x.left, x.right], list(Freqs_y.keys())))
			target_distribution = {'count': count, 'region': region}
		else:
			cont_y_num = None
			disc_y_num = 10
			target_distribution = dict(y_train.value_counts()[0:disc_y_num])
			for key in target_distribution:
				target_distribution[key] = int(target_distribution[key])
		
		shape_x_train = list(X_train.shape)
		shape_y_train = list(y_train.shape)
		if X_eval is None:
			shape_x_eval = []
		else:
			shape_x_eval = list(X_eval.shape)

		if y_eval is None:
			shape_y_eval = []
		else:
			shape_y_eval = list(y_eval.shape)

		if X_test is None:
			shape_x_test = []
		else:
			shape_x_test = list(X_test.shape)

	else:
		datatype_y = dtype2usagetype[str(y_train.dtype)]

		Missing_y = y_train.isnull().compute().tolist().count(True)
		Unique_y = len(y_train.unique().compute().tolist())

		if task == 'binary':	
			Freq_y = y_train.value_counts().compute().tolist()[0]
		else:
			Freq_y = None

		if task == 'regression':
			max_y = y_train.max().compute().tolist()
			min_y = y_train.min().compute().tolist()
			mean_y = y_train.mean().compute().tolist()
			Stdev_y = y_train.std().compute().tolist()
		else:
			max_y = None
			min_y = None
			mean_y = None
			Stdev_y = None
		
		if task == 'regression':
			Freq_y = None
			disc_y_num = None
			cont_y_num = 10
			interval = (max_y-min_y)/cont_y_num
			intervals = np.linspace(min_y, max_y, cont_y_num+1)
			Freqs_y = pd.cut(y_train, intervals).value_counts()[0:cont_y_num]
			count = list(Freqs_y)
			region = list(map(lambda x: [x.left.tolist(), x.right.tolist()], list(Freqs_y.keys())))
			target_distribution = {'count':count, 'region':region}
		else:
			cont_y_num = None
			disc_y_num = 10
			target_distribution = dict(dd.compute(y_train.value_counts())[0][0:disc_y_num])
			for key in target_distribution:
				target_distribution[key] = int(target_distribution[key])

		shape_x_train = list(X_train.shape)
		for idx, num in enumerate(shape_x_train):
			if isinstance(num, int):
				continue
			else:
				shape_x_train[idx] = num.compute()

		shape_y_train = list(map(lambda x: x.compute(), list(y_train.shape)))

		if X_eval is None:
			shape_x_eval = []
		else:
			shape_x_eval = list(X_eval.shape)
			for idx, num in enumerate(shape_x_eval):
				if isinstance(num, int):
					continue
				else:
					shape_x_eval[idx] = num.compute()
		
		if y_eval is None:
			shape_y_eval = []
		else:
			shape_y_eval = list(y_eval.shape)
			for idx, num in enumerate(shape_y_eval):
				if isinstance(num, int):
					continue
				else:
					shape_y_eval[idx] = num.compute()

		if X_test is None:
			shape_x_test = []
		else:
			shape_x_test = list(X_test.shape)
			for idx, num in enumerate(shape_x_test):
				if isinstance(num, int):
					continue
				else:
					shape_x_test[idx] = num.compute()
		
	data_character = {
		'experimentType': 'base',
		'target':{
			'name':'y',
			'taskType':task,
			'freq':Freq_y,
			'unique':Unique_y,
			'missing':Missing_y,
			'mean':mean_y, 
			'min':min_y,
			'max':max_y,
			'stdev':Stdev_y, 
			'dataType':datatype_y
		},
		'targetDistribution': target_distribution,
		'datasetShape':{
			'X_train':shape_x_train, 
			'y_train':shape_y_train, 
			'X_eval':shape_x_eval, 
			'y_eval':shape_y_eval, 
			'X_test':shape_x_test
		}
		}

	return data_character

def get_x_data_character(X_train, get_step):

	cnt_x_all = len(col_se.column_all(X_train))
	cnt_x_date = len(col_se.column_all_datetime(X_train))
	cnt_x_category = len(col_se.column_object_category_bool(X_train))
	cnt_x_num = len(col_se.column_number(X_train))

	try:
		kwargs = get_step('feature_generation').transformer_kwargs
		if kwargs['text_cols'] != None:
			cnt_x_text = len(kwargs['text_cols'])
			cnt_x_all -= len(col_se.column_text(X_train)) - len(kwargs['text_cols'])
		else:
			cnt_x_text = len(col_se.column_text(X_train))
			
		if kwargs['latlong_cols'] != None:
			cnt_x_latlong = len(kwargs['latlong_cols'])
			cnt_x_all += len(kwargs['latlong_cols'])
		else:
			cnt_x_latlong = 0
	except:
		cnt_x_text = len(col_se.column_text(X_train))
		cnt_x_latlong = 0

	cnt_x_others = cnt_x_all - cnt_x_date - cnt_x_category - cnt_x_num - cnt_x_text - cnt_x_latlong

	x_types = {
		'experimentType': 'compete',
		'featureDistribution':{
		'nContinuous':cnt_x_num, 
		'nText':cnt_x_text,
		'nDatetime':cnt_x_date,
		'nCategorical':cnt_x_category,
		'nLocation':cnt_x_latlong,
		'nOthers':cnt_x_others
		}
	}

	return x_types

