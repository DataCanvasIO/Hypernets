# -*- coding:utf-8 -*-
"""

"""

from sklearn.compose import make_column_selector
from scipy.stats import skew, kurtosis

column_object_category_bool = make_column_selector(dtype_include=['object', 'category', 'bool'])
column_object = make_column_selector(dtype_include=['object'])
column_category = make_column_selector(dtype_include=['category'])
column_bool = make_column_selector(dtype_include=['bool'])
column_number = make_column_selector(dtype_include='number')
column_number_exclude_timedelta = make_column_selector(dtype_include='number', dtype_exclude='timedelta')

column_timedelta = make_column_selector(dtype_include='timedelta')
column_datetimetz = make_column_selector(dtype_include='datetimetz')
column_datetime = make_column_selector(dtype_include='datetime')
column_all_datetime = make_column_selector(dtype_include=['datetime', 'datetimetz'])
column_int = make_column_selector(dtype_include=['int16', 'int32', 'int64'])

def column_skew_kurtosis(X, skew_threshold=0.5, kurtosis_threshold=0.5):
    column_number = column_number_exclude_timedelta(X)
    skew_values = skew(X[column_number], axis=0, nan_policy='omit')
    kurtosis_values = kurtosis(X[column_number],axis=0,nan_policy='omit')
    columns = [c for i, c in enumerate(column_number) if abs(skew_values[i]) > skew_threshold or abs(kurtosis_values[i])> kurtosis_threshold]
    return columns
