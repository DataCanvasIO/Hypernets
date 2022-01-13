# -*- coding:utf-8 -*-
"""

"""
import os
import glob
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_META_TYPE_KEY = b'hyn_type'
_META_TYPE_ARRAY = b'array'
_META_TYPE_SERIES = b'series'
_META_TYPE_DATAFRAME = b'dataframe'


class Persistence:
    def store(self, data, path, *, filesystem=None, **kwargs):
        raise NotImplemented()

    def load(self, path, *, filesystem=None, **kwargs):
        raise NotImplemented()


class ParquetPersistence(Persistence):
    acceptable_types = (np.ndarray, pd.DataFrame, pd.Series)

    def store(self, data, path, *, filesystem=None, metadata=None, **kwargs):
        """
        Store dask dataframe into parquet file(s)

        :param data: pd.DataFrame or pd.Series or np.ndarray
        :param path: the target parquet file path if df is pandas dataframe,
            or the root directory of partitioned parquet files for dask dataframe
        :param filesystem: pyarrow FileSystem or fsspec FileSystem
        :param metadata: meta dict to stored with data
        :param kwargs: options passed to pyarrow.parquet.write_table
        :return: parquet file paths tuple or delayed tasks(dask only)
        """
        assert isinstance(data, self.acceptable_types)

        if metadata is None:
            metadata = {}

        if isinstance(data, np.ndarray):
            assert len(data.shape) == 2
            metadata[_META_TYPE_KEY] = _META_TYPE_ARRAY
            data = pd.DataFrame(data, columns=map(str, range(data.shape[1])))
        elif isinstance(data, pd.Series):
            metadata[_META_TYPE_KEY] = _META_TYPE_SERIES
            data = data.to_frame()
        # else:
        #     metadata[_META_TYPE_KEY] = _META_TYPE_DATAFRAME

        result = self._arrow_store_parquet(data, path, filesystem=filesystem, metadata=metadata, **kwargs)
        return result,

    def load(self, path, *, filesystem=None, return_metadata=False, **kwargs):
        files = self._list_part_files(path, filesystem=filesystem)
        assert len(files) > 0, f'Not parquet data file at {path}'

        tbls = [pq.read_pandas(f, filesystem=filesystem, **kwargs) for f in files]
        dfs = [tbl.to_pandas() for tbl in tbls]
        df = pd.concat(dfs, axis=0) if len(dfs) > 1 else dfs[0]

        metadata = tbls[0].schema.metadata
        dtype = metadata.get(_META_TYPE_KEY, _META_TYPE_DATAFRAME)
        if dtype == _META_TYPE_ARRAY:
            result = df.values
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
    def _arrow_store_parquet(df, target_path, filesystem=None, metadata=None, **pa_options):
        assert metadata is None or isinstance(metadata, dict)

        if metadata:
            sch = pa.Schema.from_pandas(df)
            m = sch.metadata
            m.update(metadata)
            sch = sch.with_metadata(m)
            tbl = pa.Table.from_pandas(df, schema=sch)
        else:
            tbl = pa.Table.from_pandas(df)

        pq.write_table(tbl, target_path, filesystem=filesystem, **pa_options)

        return target_path

    @staticmethod
    def _list_part_files(path, filesystem=None):
        if filesystem is None:
            isdir = os.path.isdir
            glob_iter = glob.glob
            path_sep = os.path.sep
        else:
            isdir = filesystem.isdir
            glob_iter = filesystem.glob
            is_local = type(filesystem).__name__.lower().find('local') >= 0
            path_sep = os.path.sep if is_local else '/'

        if not glob.has_magic(path) and isdir(path):
            path = path.rstrip(path_sep) + f'{path_sep}*.parquet'

        if glob.has_magic(path):
            files = list(glob_iter(path))
            files.sort()
        else:
            files = [path]

        return files
