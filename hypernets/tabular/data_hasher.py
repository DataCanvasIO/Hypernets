# -*- coding:utf-8 -*-
"""

"""
import hashlib
import pickle
from io import BytesIO

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object


class DataHasher:
    def __init__(self, method='md5'):
        self.method = method

    def __call__(self, data):
        m = getattr(hashlib, self.method)()
        for x in self._iter_data(data):
            m.update(x)
        return m.hexdigest()

    def _iter_data(self, data):
        yield self._qname(type(data)).encode('utf-8')

        if isinstance(data, pd.DataFrame):
            yield from self._iter_pd_dataframe(data)
        elif isinstance(data, pd.Series):
            yield from self._iter_pd_dataframe(data.to_frame())
        elif isinstance(data, np.ndarray):
            yield from self._iter_ndarray(data)
        elif isinstance(data, (bytes, bytearray)):
            yield data
        elif isinstance(data, str):
            yield data.encode('utf-8')
        elif isinstance(data, (list, tuple)):
            for x in data:
                yield from self._iter_data(x)
        elif isinstance(data, dict):
            for k, v in data.items():
                yield from self._iter_data(k)
                yield b'='
                yield from self._iter_data(v)
        else:
            buf = BytesIO()
            pickle.dump(data, buf, protocol=pickle.HIGHEST_PROTOCOL)
            yield buf.getvalue()
            buf.close()

    @staticmethod
    def _qname(cls):
        return f'{cls.__module__}.{cls.__name__}'

    @staticmethod
    def _iter_pd_dataframe(df):
        # for col in df.columns:
        #     yield str(col).encode()
        yield ','.join(map(str, df.columns.tolist())).encode('utf-8')

        x = hash_pandas_object(df, index=False)
        yield x.values

    @staticmethod
    def _iter_ndarray(arr):
        if arr.shape[0] == 0:
            v = np.array([], dtype='u8').reshape((-1, 1))
        else:
            v = hash_pandas_object(pd.DataFrame(arr), index=False).values.reshape((-1, 1))

        yield v
