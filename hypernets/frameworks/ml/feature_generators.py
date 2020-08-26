# -*- coding:utf-8 -*-
"""

"""
import numpy as np
import featuretools as ft
from featuretools import IdentityFeature


class FeatureToolsTransformer():

    def __init__(self, trans_primitives=None, max_depth=1):
        """

        Args:
            trans_primitives:
                for continuous: "add_numeric","subtract_numeric","divide_numeric","multiply_numeric","negate","modulo_numeric","modulo_by_feature","cum_mean","cum_sum","cum_min","cum_max","percentile","absolute"
                for datetime: "year", "month", "week", "minute", "day", "hour", "minute", "second", "weekday", "is_weekend"
                for text(not support now): "num_characters", "num_words"
            max_depth:
        """
        self.trans_primitives = trans_primitives
        self.max_depth = max_depth
        self._feature_defs = None

        self._imputed = None

    def fit(self, X, **kwargs):
        # NaN,Inf,-Inf is not allowed.
        self._check_values(X)

        es = ft.EntitySet(id='es_hypernets_fit')
        es.entity_from_dataframe(entity_id='e_hypernets_ft', dataframe=X,  make_index=True, index='e_hypernets_ft_index')
        feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="e_hypernets_ft",
                                              ignore_variables={"e_hypernets_ft": []},
                                              return_variable_types="all",
                                              trans_primitives=self.trans_primitives,
                                              max_depth=self.max_depth,
                                              features_only=False,
                                              max_features=-1)
        self._feature_defs = feature_defs

        derived_cols = list(map(lambda _: _._name, filter(lambda _: not isinstance(_, IdentityFeature), feature_defs)))
        invalid_cols = self._checkout_invalid_cols(feature_matrix)

        valid_cols = set(derived_cols) - set(invalid_cols)
        self._imputed = feature_matrix[valid_cols].replace([np.inf, -np.inf], np.nan).mean()

        return self

    def transform(self, X):
        # 1. check is fitted and values
        assert self._feature_defs is not None, 'Please fit it first.'
        self._check_values(X)

        # 2. transform
        es = ft.EntitySet(id='es_hypernets_transform')
        es.entity_from_dataframe(entity_id='e_hypernets_ft', dataframe=X, make_index=False)
        feature_matrix = ft.calculate_feature_matrix(self._feature_defs, entityset=es, n_jobs=1, verbose=10)

        # 3. fill exception values
        feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_matrix.fillna(self._imputed, inplace=True)  # datetime never filled

        return feature_matrix

    def _contains_null_cols(self, df):
        _df = df.replace([np.inf, -np.inf], np.nan)
        return list(map(lambda _: _[0], filter(lambda _: _[1] > 0,  _df.isnull().sum().to_dict().items())))

    def _check_values(self, df):
        nan_cols = self._contains_null_cols(df)
        if len(nan_cols) > 0:
            _s = ",".join(nan_cols)
            raise ValueError(f"Following columns contains NaN,Inf,-Inf value that can not derivation: {_s} .")

    def _checkout_invalid_cols(self, df):
        result = []
        _df = df.replace([np.inf, -np.inf], np.nan)

        if _df.shape[0] > 0:
            for col in _df:
                if _df[col].nunique(dropna=False) < 1 or _df[col].dropna().shape[0] < 1:
                    result.append(col)
        return result
