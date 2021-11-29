# -*- coding:utf-8 -*-
"""

"""

from hypernets.utils import logging
from ..pseudo_labeling import PseudoLabeling

logger = logging.get_logger(__name__)


class CumlPseudoLabeling(PseudoLabeling):
    import cupy as np

    def _filter_by_quantile(self, proba):
        """
        cupy does not support *nanquantile*
        """
        np = self.np

        q = []
        for i in range(proba.shape[1]):
            p = proba[:, i]
            p = p[p > 0.]
            if len(p) > 0:
                q.append(np.quantile(p, self.quantile))
            else:
                q.append(1.)
        selected = (proba >= np.array(q))
        return selected
