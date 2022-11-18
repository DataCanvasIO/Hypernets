import re

import featuretools as ft
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from hypernets.tabular import get_tool_box
from hypernets.tabular.column_selector import column_all_datetime, column_number_exclude_timedelta, \
    column_category, column_bool
from hypernets.tabular.sklearn_ex import FeatureSelectionTransformer
from . import _base
from ._primitives import CrossCategorical, GeoHashPrimitive, DaskCompatibleHaversine, TfidfPrimitive
from ._primitives import is_geohash_installed

_named_primitives = [CrossCategorical, DaskCompatibleHaversine, TfidfPrimitive]

_DEFAULT_PRIMITIVES_UNKNOWN = []
_DEFAULT_PRIMITIVES_NUMERIC = []
_DEFAULT_PRIMITIVES_CATEGORY = [CrossCategorical.name]
_DEFAULT_PRIMITIVES_DATETIME = ["month", "week", "day", "hour", "minute", "second", "weekday", "is_weekend"]
# _DEFAULT_PRIMITIVES_LATLONG = [GeoHashPrimitive.name, "haversine"]
_DEFAULT_PRIMITIVES_LATLONG = [DaskCompatibleHaversine.name, ]
_DEFAULT_PRIMITIVES_TEXT = [TfidfPrimitive.name]

if is_geohash_installed:
    _DEFAULT_PRIMITIVES_LATLONG.append(GeoHashPrimitive.name)
    _named_primitives.append(GeoHashPrimitive)

_named_primitives = {p.name: p for p in _named_primitives}

# _fix_feature_name
_pattern_to_sub = re.compile(r'[=()\[\], ]')
_chars_to_replace = {'+': 'A',
                     '-': 'S',
                     '*': 'X',
                     '/': 'D',
                     '%': 'M'}


def _fix_feature_name(s):
    s = _pattern_to_sub.sub('__', s)
    for k, v in _chars_to_replace.items():
        s = s.replace(k, v)
    return s


class FeatureGenerationTransformer(BaseEstimator):
    ft_index = 'e_hypernets_ft_index'

    def __init__(self, task=None, trans_primitives=None,
                 fix_input=False,
                 categories_cols=None,
                 continuous_cols=None,
                 datetime_cols=None,
                 latlong_cols=None,
                 text_cols=None,
                 max_depth=1,
                 max_features=-1,
                 drop_cols=None,
                 feature_selection_args=None,
                 fix_feature_names=True,
                 categorical_as_object=True,
                 bool_as_int=True):
        """

        Args:
            trans_primitives:
                for categories: "cross_categorical"
                for continuous: "add_numeric","subtract_numeric","divide_numeric","multiply_numeric","negate","modulo_numeric","modulo_by_feature","cum_mean","cum_sum","cum_min","cum_max","percentile","absolute"
                for datetime: "year", "month", "week", "day", "hour", "minute", "second", "weekday", "is_weekend"
                for lat_long: "haversine", "geohash"(only if python-geohash is installed)
                for text: "num_characters", "num_words" + "tfidf"
            max_depth:
        """
        assert trans_primitives is None or isinstance(trans_primitives, (list, tuple)) and len(trans_primitives) > 0
        assert all([c is None or isinstance(c, (tuple, list))
                    for c in (categories_cols, continuous_cols, datetime_cols, latlong_cols, text_cols)])

        # if latlong_cols is not None and len(latlong_cols) > 0:
        #     assert is_geohash_installed, \
        #         f'python-geohash is required to run FeatureGeneration with latlong columns.'

        self.trans_primitives = trans_primitives
        self.max_depth = max_depth
        self.max_features = max_features
        self.task = task

        self.fix_input = fix_input
        self.categories_cols = categories_cols
        self.continuous_cols = continuous_cols
        self.datetime_cols = datetime_cols
        self.latlong_cols = latlong_cols
        self.text_cols = text_cols
        self.drop_cols = drop_cols
        self.feature_selection_args = feature_selection_args
        self.fix_feature_names = fix_feature_names
        self.categorical_as_object = categorical_as_object
        self.bool_as_int = bool_as_int

        # fitted
        self.original_cols = []
        self.selection_transformer = None
        self.imputed_input_ = None
        self.feature_defs_ = None
        self.transformed_feature_names_ = None
        self.feature_defs_names_ = None

    def fit(self, X, y=None, *, reserve_index=False, **kwargs):
        original_cols = X.columns.to_list()

        if self.feature_selection_args is not None:
            assert y is not None, '`y` must be provided for feature selection.'
            self.feature_selection_args['reserved_cols'] = original_cols
            self.selection_transformer = FeatureSelectionTransformer(task=self.task, **self.feature_selection_args)

        if self.categories_cols is None:
            self.categories_cols = []
        if self.continuous_cols is None:
            self.continuous_cols = column_number_exclude_timedelta(X)
        if self.datetime_cols is None:
            self.datetime_cols = column_all_datetime(X)
        if self.latlong_cols is None:
            self.latlong_cols = []
        if self.text_cols is None:
            self.text_cols = []

        X, y = self._fix_input(X, y, for_fit=True)

        if self.trans_primitives is None:
            self.trans_primitives = self._default_trans_primitives(X, y)

        trans_primitives = self.trans_primitives
        if any([isinstance(p, str) and p in _named_primitives.keys() for p in trans_primitives]):
            trans_primitives = [_named_primitives.get(p, p) if isinstance(p, str) else p
                                for p in trans_primitives]

        es = ft.EntitySet(id='es_hypernets_fit')
        make_index = self.ft_index not in original_cols
        feature_type_dict, _ = self._get_feature_types(X)

        if _base.FT_V0:
            es.entity_from_dataframe(entity_id='e_hypernets_ft', dataframe=X, variable_types=feature_type_dict,
                                     make_index=make_index, index=self.ft_index)
            feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="e_hypernets_ft",
                                                  ignore_variables={"e_hypernets_ft": []},
                                                  return_variable_types="all",
                                                  trans_primitives=trans_primitives,
                                                  drop_exact=self.drop_cols,
                                                  max_depth=self.max_depth,
                                                  max_features=self.max_features,
                                                  features_only=False)
            if make_index and not reserve_index:
                X.pop(self.ft_index)
        else:
            es.add_dataframe(dataframe=X.copy(), dataframe_name='e_hypernets_ft',
                             index=self.ft_index,
                             make_index=make_index, logical_types=feature_type_dict)
            feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name="e_hypernets_ft",
                                                  ignore_columns={"e_hypernets_ft": []},
                                                  return_types="all",
                                                  trans_primitives=trans_primitives,
                                                  drop_exact=self.drop_cols,
                                                  max_depth=self.max_depth,
                                                  max_features=self.max_features,
                                                  features_only=False)

        self.feature_defs_ = feature_defs
        self.original_cols = original_cols
        feature_defs_names, replaced_feature_defs_names, = self._get_transformed_feature_names(feature_defs)
        self.transformed_feature_names_ = replaced_feature_defs_names
        self.feature_defs_names_ = feature_defs_names

        if self.selection_transformer is not None:
            self.selection_transformer.fit(feature_matrix, y)
            selected_defs = []
            for fea in self.feature_defs_:
                if fea._name in self.selection_transformer.columns_:
                    selected_defs.append(fea)
            self.feature_defs_ = selected_defs

        return self

    def transform(self, X, y=None):
        # 1. check is fitted and values
        assert self.feature_defs_ is not None, 'Please fit it first.'

        # 2. fix input
        X, y = self._fix_input(X, y, for_fit=False)

        # 3. transform
        es = ft.EntitySet(id='es_hypernets_transform')
        feature_type_dict, unknown_cols = self._get_feature_types(X)
        unknown_dtypes = X[unknown_cols].dtypes.copy() if unknown_cols else {}
        make_index = self.ft_index not in X.columns.to_list()

        if _base.FT_V0:
            es.entity_from_dataframe(entity_id='e_hypernets_ft', dataframe=X, variable_types=feature_type_dict,
                                     make_index=make_index, index=self.ft_index)
        else:
            if make_index:
                tb = get_tool_box(X)
                X = tb.reset_index(X)
                X[self.ft_index] = X.index
            es.add_dataframe(dataframe=X, dataframe_name='e_hypernets_ft',
                             index=self.ft_index, make_index=False, logical_types=feature_type_dict)

        Xt = ft.calculate_feature_matrix(self.feature_defs_, entityset=es, n_jobs=1, verbose=False)
        if make_index:
            X.pop(self.ft_index)
            if self.ft_index in Xt.columns.to_list():
                Xt.pop(self.ft_index)

        # restore dtypes of unknown_cols
        for c, d in unknown_dtypes.items():
            try:
                if Xt[c].dtype != d:
                    Xt[c] = Xt[c].astype(d)
            except:
                pass

        if self.categorical_as_object:
            cat_cols = column_category(Xt)
            if cat_cols:
                Xt[cat_cols] = Xt[cat_cols].astype('object')
        if self.bool_as_int:
            bool_cols = column_bool(Xt)
            if bool_cols:
                Xt[bool_cols] = Xt[bool_cols].astype('int')
        Xt = Xt.replace([np.inf, -np.inf], np.nan)

        if self.fix_feature_names:
            Xt = self._fix_transformed_feature_names(Xt)

        return Xt

    def fit_transform(self, X, y=None, **fit_params):
        make_index = self.ft_index not in X.columns.to_list()

        self.fit(X, y, reserve_index=True, **fit_params)
        Xt = self.transform(X)

        if make_index:
            if self.ft_index in X.columns.to_list():
                X.pop(self.ft_index)
            if self.ft_index in Xt.columns.to_list():
                Xt.pop(self.ft_index)
        return Xt

    @staticmethod
    def _merge_dict(dest_dict, *dicts):
        for d in dicts:
            for k, v in d.items():
                dest_dict.setdefault(k, v)

    @property
    def classes_(self):
        if self.feature_defs_ is None:
            return None
        feats = [fea._name for fea in self.feature_defs_]
        return feats

    def _default_trans_primitives(self, X, y):
        primitives = []

        if self.categories_cols:
            primitives += _DEFAULT_PRIMITIVES_CATEGORY
        if self.continuous_cols:
            primitives += _DEFAULT_PRIMITIVES_NUMERIC
        if self.datetime_cols:
            primitives += _DEFAULT_PRIMITIVES_DATETIME
        if self.text_cols:
            primitives += _DEFAULT_PRIMITIVES_TEXT
        if self.latlong_cols:
            primitives += _DEFAULT_PRIMITIVES_LATLONG

        return primitives

    def _fix_input(self, X, y, for_fit=True):
        if self.fix_input:
            if for_fit:
                self.imputed_input_ = self._detect_impute_dict(X)
            X = self._replace_invalid_values(X, self.imputed_input_)

        return X, y

    def _detect_impute_dict(self, X):
        imputed_input = {}
        if len(self.continuous_cols) > 0:
            _mean = X[self.continuous_cols].mean()
            if hasattr(_mean, 'compute'):
                _mean = _mean.compute()
            self._merge_dict(imputed_input, _mean.to_dict())
        if len(self.datetime_cols) > 0:
            _mode = X[self.datetime_cols].mode()
            if hasattr(_mode, 'compute'):
                _mode = _mode.compute()
            if isinstance(_mode, pd.Series):
                _mode = _mode.to_frame()
            self._merge_dict(imputed_input, _mode.iloc[0].to_dict())
        if len(self.text_cols) > 0:
            self._merge_dict(imputed_input, {c: '' for c in self.text_cols})
        if len(self.latlong_cols) > 0:
            pass  # do nothing

        return imputed_input

    def _replace_invalid_values(self, df, imputed_dict):
        df = df.replace([np.inf, -np.inf], np.nan)
        if imputed_dict is not None and len(imputed_dict) > 0:
            df = df.fillna(imputed_dict)
        else:
            df = df.fillna(0)

        return df

    def _get_transformed_feature_names(self, feature_defs):
        names = [n for f in feature_defs for n in f.get_feature_names()]
        if self.fix_feature_names:
            replaced_names = [_fix_feature_name(n) for n in names]
        else:
            replaced_names = names

        return names, replaced_names

    def _fix_transformed_feature_names(self, df):
        if self.fix_feature_names and hasattr(df, 'columns'):
            columns = [_fix_feature_name(n) for n in df.columns.to_list()]
            df.columns = columns

        return df

    def _get_feature_types(self, X):
        feature_types = {}
        original_cols = X.columns.to_list()

        known_cols = self.categories_cols + self.continuous_cols + self.datetime_cols \
                     + self.latlong_cols + self.text_cols
        unknown_cols = [c for c in original_cols if c not in known_cols]
        self._merge_dict(feature_types,
                         {c: _base.Numeric for c in self.continuous_cols},
                         {c: _base.Datetime for c in self.datetime_cols},
                         {c: _base.LatLong for c in self.latlong_cols},
                         {c: _base.NaturalLanguage for c in self.text_cols},
                         {c: _base.Categorical for c in self.categories_cols},
                         {c: _base.Unknown for c in unknown_cols},
                         )
        return feature_types, unknown_cols
