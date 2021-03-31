# -*- coding:utf-8 -*-
"""

"""

import os

import dask
import pandas as pd
import pyarrow as pa
import pyarrow.filesystem as pafs
import pyarrow.parquet as pq
from dask import dataframe as dd

try:
    import fsspec

    _has_fsspec = True
except:
    _has_fsspec = False

__all__ = ('to_parquet', 'read_parquet')

pa_ensure_filesystem = pafs._ensure_filesystem


def _check_pa_filesystem(filesystem):
    if hasattr(filesystem, '_hyn_adapted_'):
        # print('*' * 20, 'return adapted fs')
        return filesystem

    # print('-' * 20, 'call arrow _ensure_filesystem')
    return pa_ensure_filesystem(filesystem)


pafs._ensure_filesystem = _check_pa_filesystem


def _check_filesystem(filesystem):
    if _has_fsspec:
        if isinstance(filesystem, fsspec.AbstractFileSystem):
            try:
                # do this line here to keep LocalFileSystem as fsspec
                filesystem = pa.fs.PyFileSystem(pa.fs.FSSpecHandler(filesystem))
            except:
                pass

    return filesystem


def _arrow_write_parquet(df, target_path, filesystem=None, **pa_options):
    filesystem = _check_filesystem(filesystem)
    tbl = pa.Table.from_pandas(df)
    pq.write_table(tbl, target_path, filesystem=filesystem, **pa_options)

    return target_path


def to_parquet(df, path, filesystem=None, delayed=False, **kwargs_pass):
    """
    Use pyarrow to store pandas or dask dataframe into parquet file(s)

    :param df: a pandas or dask dataframe instance
    :param path: the target parquet file path if df is pandas dataframe,
        or the root directory of partitioned parquet files for dask dataframe
    :param filesystem: pyarrow FileSystem or fsspec FileSystem
    :param delayed: [dask only]
    :param kwargs_pass: options passed to pyarrow.parquet.write_table
    :return: parquet file paths tuple or delayed tasks(dask only)
    """
    assert isinstance(df, (pd.DataFrame, dd.DataFrame))

    if isinstance(df, pd.DataFrame):
        result = _arrow_write_parquet(df, path, filesystem, **kwargs_pass)
        return (result,)

    # write dask dataframe

    is_local = type(filesystem).__name__.lower().find('local') >= 0 if filesystem else True
    path_sep = os.path.sep if is_local else '/'

    path = path.rstrip(path_sep)
    filenames = ["part.%i.parquet" % (i) for i in range(df.npartitions)]
    delayed_write = dask.delayed(_arrow_write_parquet)
    parts = [
        delayed_write(d, f'{path}/{filename}', filesystem, **kwargs_pass)
        for d, filename in zip(df.to_delayed(), filenames)
    ]

    if delayed:
        return parts
    else:
        result = dask.compute(parts)
        return result


def read_parquet(path, delayed=False, **kwargs_pass):
    if delayed:
        if 'filesystem' in kwargs_pass:
            filesystem = kwargs_pass.pop('filesystem')
            assert _has_fsspec and isinstance(filesystem, fsspec.AbstractFileSystem)

            return _adapted_dask_read_parquet(path, fs=filesystem, **kwargs_pass)
        else:
            return dd.read_parquet(path, **kwargs_pass)
    else:
        if 'filesystem' in kwargs_pass and _has_fsspec:
            filesystem = kwargs_pass['filesystem']
            if isinstance(filesystem, fsspec.AbstractFileSystem):
                try:
                    # do this line here to keep LocalFileSystem as fsspec
                    filesystem = pa.fs.PyFileSystem(pa.fs.FSSpecHandler(filesystem))
                    kwargs_pass['filesystem'] = filesystem
                except:
                    pass
        return pd.read_parquet(path, **kwargs_pass)


def _adapted_dask_read_parquet(
        path,
        fs,
        columns=None,
        filters=None,
        categories=None,
        index=None,
        storage_options=None,
        engine="auto",
        gather_statistics=None,
        split_row_groups=None,
        chunksize=None,
        **kwargs, ):
    """
    Adapted form dask.dataframe.read_parquet with given filesystem.
    {}
    """.format(dd.read_parquet.__doc__)

    from dask.dataframe.io.parquet.core \
        import tokenize, natural_sort_key, process_statistics, \
        set_index_columns, NONE_LABEL, ParquetSubgraph, new_dd_object

    if isinstance(columns, str):
        df = _adapted_dask_read_parquet(
            path,
            [columns],
            filters,
            categories,
            index,
            storage_options,
            engine,
            gather_statistics,
        )
        return df[columns]

    if columns is not None:
        columns = list(columns)

    name = "read-parquet-" + tokenize(
        path,
        columns,
        filters,
        categories,
        index,
        storage_options,
        engine,
        gather_statistics,
    )

    if isinstance(engine, str):
        engine = dd.io.parquet.core.get_engine(engine)

    if hasattr(path, "name"):
        path = dd.io.parquet.core.stringify_path(path)

    # == adapt start ==
    # fs, _, paths = get_fs_token_paths(path, mode="rb", storage_options=storage_options)
    if "*" in path:
        paths = [f for f in sorted(fs.glob(path)) if not fs.isdir(f)]
    else:
        paths = [path]
    # == adapt end ==

    paths = sorted(paths, key=natural_sort_key)  # numeric rather than glob ordering

    auto_index_allowed = False
    if index is None:
        # User is allowing auto-detected index
        auto_index_allowed = True
    if index and isinstance(index, str):
        index = [index]

    # if type(fs).__name__ == 'S3FileSystem':
    #     fs = AdaptedS3FSWrapper(fs)
    meta, statistics, parts, index = engine.read_metadata(
        fs,
        paths,
        categories=categories,
        index=index,
        gather_statistics=gather_statistics,
        filters=filters,
        split_row_groups=split_row_groups,
        **kwargs,
    )

    # Parse dataset statistics from metadata (if available)
    parts, divisions, index, index_in_columns = process_statistics(
        parts, statistics, filters, index, chunksize
    )

    # Account for index and columns arguments.
    # Modify `meta` dataframe accordingly
    meta, index, columns = set_index_columns(
        meta, index, columns, index_in_columns, auto_index_allowed
    )
    if meta.index.name == NONE_LABEL:
        meta.index.name = None

    subgraph = ParquetSubgraph(name, engine, fs, meta, columns, index, parts, kwargs)

    # Set the index that was previously treated as a column
    if index_in_columns:
        meta = meta.set_index(index)
        if meta.index.name == NONE_LABEL:
            meta.index.name = None

    if len(divisions) < 2:
        # empty dataframe - just use meta
        subgraph = {(name, 0): meta}
        divisions = (None, None)

    return new_dd_object(subgraph, name, meta, divisions)


class AdaptedS3FSWrapper(pa.filesystem.S3FSWrapper):

    def walk(self, path, refresh=False):
        # """
        # Directory tree generator, like os.walk.
        #
        # Generator version of what is in s3fs, which yields a flattened list of
        # files.
        # """
        # path = _sanitize_s3(_stringify_path(path))
        # directories = set()
        # files = set()
        #
        # for key in list(self.fs._ls(path, refresh=refresh)):
        #     path = key['Key']
        #     if key['StorageClass'] == 'DIRECTORY':
        #         directories.add(path)
        #     elif key['StorageClass'] == 'BUCKET':
        #         pass
        #     else:
        #         files.add(path)
        #
        # # s3fs creates duplicate 'DIRECTORY' entries
        # files = sorted([posixpath.split(f)[1] for f in files
        #                 if f not in directories])
        # directories = sorted([posixpath.split(x)[1]
        #                       for x in directories])
        #
        # yield path, directories, files
        #
        # for directory in directories:
        #     yield from self.walk(directory, refresh=refresh)
        yield from self.fs.walk(path, refresh=refresh)
