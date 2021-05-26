# coding: UTF-8
"""

"""
from functools import partial

import numpy as np
from featuretools import variable_types, primitives

from hypernets.tabular.cfg import TabularCfg as cfg

try:
    import geohash

    _installed_geohash = True
except ImportError:
    _installed_geohash = False


class CrossCategorical(primitives.TransformPrimitive):
    name = "cross_categorical"
    input_types = [variable_types.Categorical, variable_types.Categorical]
    return_type = variable_types.Categorical
    commutative = True
    dask_compatible = True

    def get_function(self):
        return self.char_add

    def char_add(self, x1, x2):
        return np.char.add(np.array(x1, 'U'), np.char.add('__', np.array(x2, 'U')))

    def generate_name(self, base_feature_names):
        return "%s__%s" % (base_feature_names[0], base_feature_names[1])


class GeoHashPrimitive(primitives.TransformPrimitive):
    name = "geohash"
    input_types = [variable_types.LatLong]
    return_type = variable_types.Categorical
    commutative = True
    dask_compatible = True

    def __init__(self, precision=cfg.geohash_precision):
        assert _installed_geohash, f'Failed to import geohash, "python-geohash" is required.'

        super(GeoHashPrimitive, self).__init__()

        self.precision = precision

    def get_function(self):
        fn = partial(self._geo_hash, precision=self.precision)
        vfn = np.vectorize(fn, otypes=[np.object], signature='()->()')
        return vfn

    @staticmethod
    def _geo_hash(x, precision=12):
        return geohash.encode(x[0], x[1], precision=precision)
