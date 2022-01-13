# -*- coding:utf-8 -*-
"""

"""
import os

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pyarrow.parquet as pq

from ..persistence import ParquetPersistence
from ..persistence import _META_TYPE_KEY, _META_TYPE_DATAFRAME, _META_TYPE_SERIES, _META_TYPE_ARRAY


class DaskParquetPersistence(ParquetPersistence):
    acceptable_types = ParquetPersistence.acceptable_types + (dd.DataFrame, dd.Series, da.Array)

    def store(self, data, path, *, filesystem=None, metadata=None, delayed=False, **kwargs):
        assert isinstance(data, self.acceptable_types)

        if isinstance(data, ParquetPersistence.acceptable_types):
            return super(DaskParquetPersistence, self).store(data, path, filesystem=filesystem, **kwargs)

        # check path
        is_local = type(filesystem).__name__.lower().find('local') >= 0 if filesystem else True
        path_sep = os.path.sep if is_local else '/'
        path = path.rstrip(path_sep)
        if filesystem is None:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
        else:
            if not filesystem.exists(path):
                filesystem.mkdirs(path, exist_ok=True)

        # data to dataframe, and get metadata
        df, metadata = self._to_df(data, metadata)

        # write dataframe
        delayed_write = dask.delayed(ParquetPersistence._arrow_store_parquet)
        filenames = self._get_part_filenames(df.npartitions)
        parts = [
            delayed_write(d, f'{path}/{filename}', filesystem=filesystem, metadata=metadata, **kwargs)
            for d, filename in zip(df.to_delayed(), filenames)
        ]

        # return
        if delayed:
            return parts
        else:
            result = dask.compute(*parts)
            return result

    def load(self, path, *, filesystem=None, return_metadata=False, delayed=True, **kwargs):
        if not delayed:
            return super().load(path, filesystem=filesystem, **kwargs)

        files = self._list_part_files(path, filesystem=filesystem)
        assert len(files) > 0, f'Not found data: {path}'

        # get schema and meta from first partition
        tbl = pq.read_table(files[0], filesystem=filesystem)
        schema = tbl.schema
        metadata = schema.metadata
        # df_meta = self._get_dataframe_meta(schema)
        dtype = metadata.get(_META_TYPE_KEY, _META_TYPE_DATAFRAME)

        # load dataframe as delayed
        delayed_load = dask.delayed(type(self)._arrow_load_parquet)
        parts = [delayed_load(f, filesystem=filesystem, **kwargs) for f in files]
        df = dd.from_delayed(parts, prefix='load_parquet', meta=None)

        # to result
        if dtype == _META_TYPE_ARRAY:
            result = df.to_dask_array()
        elif dtype == _META_TYPE_SERIES:
            assert len(df.columns) == 1
            result = df[df.columns[0]]
        else:
            result = df

        if return_metadata:
            return result, metadata
        else:
            return result

    @staticmethod
    def _to_df(data, metadata):
        if metadata is None:
            metadata = {}

        if isinstance(data, da.Array):
            assert len(data.shape) == 2
            if any(np.nan in d for d in data.chunks):
                data = data.compute_chunk_sizes()
            df = dd.from_dask_array(data, columns=list(map(str, range(data.shape[1]))))
            metadata[_META_TYPE_KEY] = _META_TYPE_ARRAY
        elif isinstance(data, dd.Series):
            df = data.to_frame()
            metadata[_META_TYPE_KEY] = _META_TYPE_SERIES
        else:
            df = data
            # metadata[_META_TYPE_KEY] = _META_TYPE_DATAFRAME
        return df, metadata

    @staticmethod
    def _get_part_filenames(npartitions):
        return ["part.%05i.parquet" % i for i in range(npartitions)]

    @staticmethod
    def _arrow_load_parquet(path, filesystem=None, **kwargs):
        tbl = pq.read_pandas(path, filesystem=filesystem, **kwargs)
        df = tbl.to_pandas()
        return df

    @staticmethod
    def _get_dataframe_meta(schema):
        try:
            meta = schema.pandas_metadata
            meta = {c['name']: c['numpy_type'] for c in meta['columns'] if c['name'] is not None}
            return None
        except:
            return None
