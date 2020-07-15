# -*- coding:utf-8 -*-
"""

"""

from sklearn.compose import make_column_selector
from scipy.stats import skew

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


def column_skewed(X, skew_threshold=0.5):
    column_number = column_number_exclude_timedelta(X)
    skewed = X[column_number].apply(lambda x: skew(x.dropna()))
    columns = list(skewed[abs(skewed) > skew_threshold].index)
    return columns
