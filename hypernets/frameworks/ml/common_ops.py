# -*- coding:utf-8 -*-
"""

"""

from .column_selector import *
from .preprocessing import *
import numpy as np
from hypernets.core.search_space import Choice


def categorical_pipeline(impute_strategy='constant', seq_no=0):
    # impute_strategy = Choice(['constant', 'most_frequent'])  # mean,median can only be used with numeric data.

    pipeline = Pipeline([
        SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'categorical_imputer_{seq_no}'),
        MultiLabelEncoder(name=f'categorical_label_encoder_{seq_no}')
    ], columns=column_object_category_bool)
    return pipeline


def numeric_pipeline(impute_strategy='mean', seq_no=0):
    # impute_strategy = Choice(['mean', 'median', 'constant', 'most_frequent'])
    pipeline = Pipeline([
        SimpleImputer(missing_values=np.nan, strategy=impute_strategy, name=f'numeric_imputer_{seq_no}'),
        StandardScaler(name=f'numeric_standard_scaler_{seq_no}')
    ], columns=column_number_exclude_timedelta)
    return pipeline
