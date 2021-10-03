# -*- coding:utf-8 -*-
import numpy as np
from dask import array as da
from dask import dataframe as dd
from scipy import sparse as _sparse

from hypernets.tabular.dataframe_mapper import DataFrameMapper
from hypernets.utils import logging

logger = logging.get_logger(__name__)


class DaskDataFrameMapper(DataFrameMapper):
    @staticmethod
    def _fix_feature(fea):
        from ._toolbox import DaskToolBox

        if DaskToolBox.is_dask_object(fea):
            pass
        elif _sparse.issparse(fea):
            fea = fea.toarray()

        if len(fea.shape) == 1:
            """
            Convert 1-dimensional arrays to 2-dimensional column vectors.
            """
            if isinstance(fea, da.Array):
                fea = da.stack([fea], axis=-1)
            else:
                fea = np.array([fea]).T

        return fea

    @staticmethod
    def _hstack_array(extracted):
        from ._toolbox import DaskToolBox

        if DaskToolBox.exist_dask_object(*extracted):
            extracted = [a.values if isinstance(a, dd.DataFrame) else a for a in extracted]
            stacked = DaskToolBox.hstack_array(extracted)
        else:
            stacked = np.hstack(extracted)
        return stacked

    def _to_df(self, X, extracted, columns):
        if isinstance(X, dd.DataFrame):
            from ._toolbox import DaskToolBox

            dfs = [dd.from_dask_array(arr, index=None) if isinstance(arr, da.Array) else arr for arr in extracted]
            df = DaskToolBox.concat_df(dfs, axis=1) if len(dfs) > 1 else dfs[0]
            df.columns = columns
        else:
            df = super()._to_df(X, extracted, columns)

        return df
