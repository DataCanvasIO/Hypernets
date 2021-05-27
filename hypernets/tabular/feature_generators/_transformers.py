import featuretools as ft
import numpy as np
import pandas as pd
from featuretools import variable_types
from sklearn.base import BaseEstimator, TransformerMixin

from hypernets.tabular.column_selector import column_all_datetime, column_number_exclude_timedelta
from hypernets.tabular.sklearn_ex import FeatureSelectionTransformer
from ._primitives import CrossCategorical, GeoHashPrimitive

_named_primitives = [CrossCategorical, GeoHashPrimitive]

_DEFAULT_PRIMITIVES_UNKNOWN = []
_DEFAULT_PRIMITIVES_NUMERIC = []
_DEFAULT_PRIMITIVES_CATEGORY = [CrossCategorical.name]
_DEFAULT_PRIMITIVES_DATETIME = ["month", "week", "day", "hour", "minute", "second", "weekday", "is_weekend"]
_DEFAULT_PRIMITIVES_LATLONG = ["haversine", GeoHashPrimitive.name]
_DEFAULT_PRIMITIVES_TEXT = []

try:
    import nlp_primitives

    _DEFAULT_PRIMITIVES_TEXT += [nlp_primitives.LSA]
    _named_primitives += [nlp_primitives.LSA]
except ImportError:
    pass

_named_primitives = {p.name: p for p in _named_primitives}


class FeatureGenerationTransformer(BaseEstimator, TransformerMixin):
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
                 feature_selection_args=None):
        """

        Args:
            trans_primitives:
                for categories: "cross_categorical"
                for continuous: "add_numeric","subtract_numeric","divide_numeric","multiply_numeric","negate","modulo_numeric","modulo_by_feature","cum_mean","cum_sum","cum_min","cum_max","percentile","absolute"
                for datetime: "year", "month", "week", "day", "hour", "minute", "second", "weekday", "is_weekend"
                for lat_long: "haversine", "geohash"
                for text: "num_characters", "num_words" + nlp_primitives
            max_depth:
        """
        assert trans_primitives is None or isinstance(trans_primitives, (list, tuple)) and len(trans_primitives) > 0
        assert all([c is None or isinstance(c, (tuple, list))
                    for c in (categories_cols, continuous_cols, datetime_cols, latlong_cols, text_cols)])

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

        # fitted
        self._imputed_input = None
        self.original_cols = []
        self.selection_transformer = None
        self.feature_defs_ = None

    def fit(self, X, y=None, **kwargs):
        original_cols = X.columns.to_list()

        if self.feature_selection_args is not None:
            assert y is not None, '`y` must be provided for feature selection.'
            self.feature_selection_args['reserved_cols'] = original_cols
            self.selection_transformer = FeatureSelectionTransformer(task=self.task, **self.feature_selection_args)

        # self._check_values(X)
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

        known_cols = self.categories_cols + self.continuous_cols + self.datetime_cols + \
                     self.latlong_cols + self.text_cols
        unknown_cols = [c for c in original_cols if c not in known_cols]

        if self.fix_input:
            _mean = X[self.continuous_cols].mean().to_dict()
            _mode = X[self.datetime_cols].mode().to_dict()
            self._imputed_input = {}
            self._merge_dict(self._imputed_input, _mean, _mode)
            self._replace_invalid_values(X, self._imputed_input)

        feature_type_dict = {}
        self._merge_dict(feature_type_dict,
                         {c: variable_types.Numeric for c in self.continuous_cols},
                         {c: variable_types.Datetime for c in self.datetime_cols},
                         {c: variable_types.LatLong for c in self.latlong_cols},
                         {c: variable_types.NaturalLanguage for c in self.text_cols},
                         {c: variable_types.Categorical for c in self.categories_cols},
                         {c: variable_types.Unknown for c in unknown_cols},
                         )

        if self.trans_primitives is not None:
            trans_primitives = self.trans_primitives
        else:
            trans_primitives = self._default_trans_primitives(X, y)
        if any([isinstance(p, str) and p in _named_primitives.keys() for p in trans_primitives]):
            trans_primitives = [_named_primitives.get(p, p) if isinstance(p, str) else p
                                for p in trans_primitives]

        make_index = True
        if self.ft_index in original_cols:
            make_index = False

        es = ft.EntitySet(id='es_hypernets_fit')
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
        X.pop(self.ft_index)

        self.feature_defs_ = feature_defs
        self.original_cols = original_cols

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
        if self.fix_input:
            self._replace_invalid_values(X, self._imputed_input)

        # 3. transform
        es = ft.EntitySet(id='es_hypernets_transform')
        es.entity_from_dataframe(entity_id='e_hypernets_ft', dataframe=X, make_index=(self.ft_index not in X),
                                 index=self.ft_index)
        feature_matrix = ft.calculate_feature_matrix(self.feature_defs_, entityset=es, n_jobs=1, verbose=10)
        feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)

        return feature_matrix

    def _filter_by_type(self, fields, types):
        result = []
        for f, t in fields:
            for _t in types:
                if t.type == _t:
                    result.append(f)
        return result

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

    def _replace_invalid_values(self, df: pd.DataFrame, imputed_dict):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(imputed_dict, inplace=True)

    def _contains_null_cols(self, df):
        _df = df.replace([np.inf, -np.inf], np.nan)
        return list(map(lambda _: _[0], filter(lambda _: _[1] > 0, _df.isnull().sum().to_dict().items())))

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
