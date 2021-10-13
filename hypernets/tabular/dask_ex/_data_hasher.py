# -*- coding:utf-8 -*-
"""

"""
import dask.dataframe as dd
import dask.array as da
from pandas.util import hash_pandas_object

from ..data_hasher import DataHasher


class DaskDataHasher(DataHasher):

    def _iter_data(self, data):
        if isinstance(data, dd.DataFrame):
            yield from self._iter_dask_dataframe(data)
        elif isinstance(data, dd.Series):
            yield from self._iter_dask_dataframe(data.to_frame())
        elif isinstance(data, da.array):
            yield from self._iter_dask_array(data)
        else:
            yield from super()._iter_data(data)

    @staticmethod
    def _iter_dask_dataframe(df):
        yield ','.join(map(str, df.columns.tolist())).encode('utf-8')

        x = df.map_partitions(lambda part: hash_pandas_object(part, index=False), meta=(None, 'u8')).compute()
        yield x.values

    @staticmethod
    def _iter_dask_array(arr):
        if len(arr.shape) == 1:
            arr = arr.compute_chunk_sizes().reshape(-1, 1)
        x = arr.map_blocks(DataHasher._iter_ndarray, dtype='u8').compute()
        yield x
