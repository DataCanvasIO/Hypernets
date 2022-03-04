# -*- coding:utf-8 -*-
"""

"""
import cupy
import cudf

from ..data_hasher import DataHasher


class CumlDataHasher(DataHasher):

    def _iter_data(self, data):
        if isinstance(data, cudf.DataFrame):
            yield from self._iter_cudf_dataframe(data)
        elif isinstance(data, cudf.Series):
            yield from self._iter_cudf_dataframe(data.to_frame())
        elif isinstance(data, cupy.ndarray):
            yield from self._iter_cudf_dataframe(cudf.DataFrame(data), yield_columns=False)
        else:
            yield from super()._iter_data(data)

    @staticmethod
    def _iter_cudf_dataframe(df, yield_columns=True):
        if yield_columns:
            yield ','.join(map(str, df.columns.tolist())).encode('utf-8')

        if hasattr(df, 'hash_columns'):
            hashed = df.hash_columns()
        else:
            hashed = df.hash_values().values
        # hashed = cudf.DataFrame(hashed).T.hash_columns()
        yield cupy.asnumpy(hashed)
