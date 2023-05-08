# -*- coding:utf-8 -*-
"""

"""
# from ._primitives import CrossCategorical, GeoHashPrimitive, DaskCompatibleHaversine, TfidfPrimitive
# from ._transformers import FeatureGenerationTransformer, is_geohash_installed

try:
    from ._transformers import FeatureGenerationTransformer, is_geohash_installed

    is_feature_generator_ready = True
except ImportError as e:
    _msg = f'{e}, install featuretools and try again'

    is_geohash_installed = False
    is_feature_generator_ready = False

    from sklearn.base import BaseEstimator as _BaseEstimator


    class FeatureGenerationTransformer(_BaseEstimator):
        def __init__(self, *args, **kwargs):
            raise ImportError(_msg)
