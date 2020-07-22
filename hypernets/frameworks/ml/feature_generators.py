# -*- coding:utf-8 -*-
"""

"""

from .transformers import HyperTransformer
import pandas as pd
import numpy as np
import featuretools as ft


class FeatureToolsTransformer():
    def __init__(self, aggr_primitives=None, trans_primitives=None):
        self.aggr_primitives = aggr_primitives
        self.trans_primitives = trans_primitives
        self.entity_set = None
        self.features = None

    def _build_entity_set(self, X):
        entity_set = ft.EntitySet(id='entity_set')
        entity_set.entity_from_dataframe(entity_id='ft_entity', dataframe=X, make_index=True, index='ft_index')
        return entity_set

    def fit(self, X):
        self.entity_set = self._build_entity_set(X)
        self.features = ft.dfs(entityset=self.entity_set, target_entity="data",
                               # ignore_variables={'data': ignore_cols},
                               agg_primitives=self.aggr_primitives,
                               trans_primitives=self.trans_primitives,
                               # trans_primitives=self.primitives,
                               # max_depth=self.max_depth
                               features_only=True
                               )
        return self

    def transform(self, X):
        assert self.features is not None

        if self.entity_set is None:
            self.entity_set = self._build_entity_set(X)
        feature_matrix = ft.calculate_feature_matrix(self.features, self.entity_set)
        return feature_matrix

    def predict(self, X):
        return self.transform(X)
