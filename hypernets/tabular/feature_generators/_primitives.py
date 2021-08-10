# coding: UTF-8
"""

"""
from functools import partial

import geohash
import numpy as np
from featuretools import variable_types, primitives
from featuretools.primitives import Haversine
from sklearn.decomposition import TruncatedSVD as sk_TruncatedSVD
from sklearn.pipeline import make_pipeline

from hypernets.tabular import dask_ex as dex, sklearn_ex as skex
from hypernets.tabular.cfg import TabularCfg as cfg


class DaskCompatibleTransformPrimitive(primitives.TransformPrimitive):
    compatibility = [primitives.Library.PANDAS, primitives.Library.DASK]
    return_dtype = 'object'
    commutative = True

    def get_function(self):
        return self.fn_pd_or_dask

    def fn_pd_or_dask(self, x1, *args):
        if dex.is_dask_series(x1):
            if len(args) > 0:
                dfs = [x.to_frame() for x in [x1, *args]]
                df = dex.concat_df(dfs, axis=1)
            else:
                df = x1.to_frame()
            result = df.map_partitions(self.fn_dask_part, meta=(None, self.return_dtype))
            return result
        else:
            return self.fn_pd(x1, *args)

    def fn_pd(self, x1, *args):
        raise NotImplementedError()

    def fn_dask_part(self, part):
        arrs = [part.iloc[:, i] for i in range(len(part.columns))]
        return self.fn_pd(*arrs)


class CrossCategorical(DaskCompatibleTransformPrimitive):
    name = "cross_categorical"
    input_types = [variable_types.Categorical, variable_types.Categorical]
    return_type = variable_types.Categorical

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
    input_types = [variable_types.LatLong]
    return_type = variable_types.Categorical

    precision = cfg.geohash_precision

    def fn_pd(self, x1, *args):
        vfn_ = np.vectorize(partial(_geo_hash, precision=self.precision),
                            otypes=[np.object], signature='()->()')
        return vfn_(x1)


class TfidfPrimitive(primitives.TransformPrimitive):
    name = 'tfidf'
    input_types = [variable_types.NaturalLanguage]
    return_dtype = variable_types.Numeric
    commutative = True
    compatibility = [primitives.Library.PANDAS, primitives.Library.DASK]

    number_output_features = cfg.tfidf_primitive_output_feature_count
    tfidf_max_features = cfg.tfidf_max_feature_count

    def get_function(self):
        return self.fn_pd_or_dask

    def fn_pd_or_dask(self, x1):
        if isinstance(x1, dex.dd.Series):
            p = make_pipeline(dex.LocalizedTfidfVectorizer(max_features=self.tfidf_max_features),
                              dex.TruncatedSVD(n_components=self.number_output_features))
        else:
            p = make_pipeline(skex.LocalizedTfidfVectorizer(max_features=self.tfidf_max_features),
                              sk_TruncatedSVD(n_components=self.number_output_features))

        xt = p.fit_transform(x1)
        if hasattr(xt, 'iloc'):
            result = [xt.iloc[:, i] for i in range(xt.shape[1])]
        else:
            result = [xt[:, i] for i in range(xt.shape[1])]

        return result
