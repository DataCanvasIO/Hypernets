# -*- coding:utf-8 -*-
"""

"""

from .column_selector import *
from .preprocessing import *
import numpy as np
from hypernets.core.search_space import Choice


def categorical_pipeline(impute_strategy='constant'):
    impute_strategy = Choice(['constant', 'most_frequent'])  # mean,median can only be used with numeric data.

    pipeline = Pipeline([
        SimpleImputer(missing_values=np.nan, strategy='constant', name='categorical_imputer'),
        MultiLabelEncoder(name='categorical_label_encoder')
    ], columns=column_object_category_bool)
    return pipeline
