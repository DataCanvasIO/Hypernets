# coding: UTF-8
"""

"""
from functools import partial

import featuretools as ft
import numpy as np
import pandas as pd
from featuretools.primitives import Haversine, TransformPrimitive
from featuretools.utils.gen_utils import Library
from sklearn.pipeline import make_pipeline

from hypernets.tabular.cfg import TabularCfg as cfg
from hypernets.utils import Version
from . import _base

try:
    import geohash

    is_geohash_installed = True
except ImportError:
    is_geohash_installed = False

_TO_DASK_SERIES = Version(ft.__version__) >= Version('1.20')


class DaskCompatibleTransformPrimitive(TransformPrimitive):
    compatibility = [Library.PANDAS, Library.DASK]
    return_dtype = 'object'
    commutative = True

    def get_function(self):
        return self.fn_pd_or_dask

    def fn_pd_or_dask(self, x1, *args):
        from hypernets.tabular import is_dask_installed
        if is_dask_installed:
            from hypernets.tabular.dask_ex import DaskToolBox
            if DaskToolBox.is_dask_series(x1):
                if len(args) > 0:
                    dfs = [x.to_frame() for x in [x1, *args]]
                    df = DaskToolBox.concat_df(dfs, axis=1)
                else:
                    df = x1.to_frame()
                result = df.map_partitions(self.fn_dask_part, meta=(None, self.return_dtype))
                return result

        return self.fn_pd(x1, *args)

    def fn_pd(self, x1, *args):
        raise NotImplementedError()

    def fn_dask_part(self, part):
        arrs = [part.iloc[:, i] for i in range(len(part.columns))]
        result = self.fn_pd(*arrs)

        if _TO_DASK_SERIES and isinstance(result, np.ndarray):
            result = pd.Series(result)

        return result


class CrossCategorical(DaskCompatibleTransformPrimitive):
    name = "cross_categorical"
    input_types = [_base.ColumnSchema(logical_type=_base.Categorical),
                   _base.ColumnSchema(logical_type=_base.Categorical)]
    return_type = _base.ColumnSchema(logical_type=_base.Categorical, semantic_tags={'category'})

    def fn_pd(self, x1, *args):
        result = np.array(x1, 'U')
        for x in args:
            result = np.char.add(result, np.char.add('__', np.array(x, 'U')))
        return result

    def generate_name(self, base_feature_names):
        # return "%s__%s" % (base_feature_names[0], base_feature_names[1])
        s = '__'.join(base_feature_names)
        return f'{self.name.upper()}_{s}'


class DaskCompatibleHaversine(DaskCompatibleTransformPrimitive):
    stub = Haversine(unit='kilometers')

    name = f'{stub.name}_'
    input_types = stub.input_types
    return_type = stub.return_type

    def fn_pd(self, x1, *args):
        return self.stub.get_function()(x1, args[0])

    def generate_name(self, base_feature_names):
        return self.stub.generate_name(base_feature_names)


def _geo_hash(x, precision=12):
    if any(np.isnan(x)):
        return np.nan
    else:
        return geohash.encode(x[0], x[1], precision=precision)


class GeoHashPrimitive(DaskCompatibleTransformPrimitive):
    name = "geohash"
    input_types = [_base.ColumnSchema(logical_type=_base.LatLong)]
    return_type = _base.ColumnSchema(logical_type=_base.Categorical, semantic_tags={'category'})

    precision = cfg.geohash_precision

    def fn_pd(self, x1, *args):
        vfn_ = np.vectorize(partial(_geo_hash, precision=self.precision),
                            otypes=[object], signature='()->()')
        return vfn_(x1)


class TfidfPrimitive(TransformPrimitive):
    name = 'tfidf'
    input_types = [_base.ColumnSchema(logical_type=_base.NaturalLanguage)]
    return_dtype = _base.ColumnSchema(logical_type=_base.Numeric, semantic_tags={'numeric'})
    commutative = True
    compatibility = [Library.PANDAS, Library.DASK]

    def __init__(self):
        super().__init__()

        self.encoder_ = None

    @property
    def number_output_features(self):
        return cfg.tfidf_primitive_output_feature_count

    @property
    def tfidf_max_features(self):
        return cfg.tfidf_max_feature_count

    def get_function(self):
        return self.fn_pd_or_dask

    def fn_pd_or_dask(self, x1):
        if self.encoder_ is None:
            from hypernets.tabular import get_tool_box
            tfs = get_tool_box(x1).transformers
            encoder = make_pipeline(tfs['LocalizedTfidfVectorizer'](max_features=self.tfidf_max_features),
                                    tfs['TruncatedSVD'](n_components=self.number_output_features))
            xt = encoder.fit_transform(x1)
            self.encoder_ = encoder
        else:
            xt = self.encoder_.transform(x1)

        if _TO_DASK_SERIES:
            from hypernets.tabular import is_dask_installed
            if is_dask_installed:
                from hypernets.tabular.dask_ex import DaskToolBox
                if DaskToolBox.is_dask_array(xt):
                    xt = DaskToolBox.to_dask_frame_or_series(xt)
                    xt = DaskToolBox.make_divisions_known(xt)

        if hasattr(xt, 'iloc'):
            result = [xt.iloc[:, i] for i in range(xt.shape[1])]
        else:
            result = [xt[:, i] for i in range(xt.shape[1])]

        return result
