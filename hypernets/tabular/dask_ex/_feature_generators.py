# -*- coding:utf-8 -*-
"""

"""

from ..feature_generators import FeatureGenerationTransformer
from ..feature_generators import is_feature_generator_ready as _is_feature_generator_ready

is_feature_generator_ready = _is_feature_generator_ready


class DaskFeatureGenerationTransformer(FeatureGenerationTransformer):
    def _fix_input(self, X, y, for_fit=True):
        from ._toolbox import DaskToolBox

        X, y = super()._fix_input(X, y, for_fit=for_fit)
        X, y = [DaskToolBox.make_divisions_known(t) if DaskToolBox.is_dask_object(t) else t for t in (X, y)]

        return X, y
