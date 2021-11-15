# -*- coding:utf-8 -*-
"""

"""

import os
from distutils.version import LooseVersion

import dask
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dask import dataframe as dd

__all__ = ('to_parquet', 'read_parquet')


def _arrow_write_parquet(df, target_path, filesystem=None, **pa_options):
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
        result = dask.compute(*parts)
        return result


def read_parquet(path, delayed=False, **kwargs_pass):
    if delayed:
        if 'filesystem' in kwargs_pass:
            filesystem = kwargs_pass.pop('filesystem')

            if LooseVersion(dask.__version__) >= LooseVersion('2021.5.0'):
                return _adapted_dask_read_parquet_20210500(path, fs=filesystem, **kwargs_pass)
            else:
                return _adapted_dask_read_parquet(path, fs=filesystem, **kwargs_pass)
        else:
            return dd.read_parquet(path, **kwargs_pass)
    else:
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


def _adapted_dask_read_parquet_20210500(
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
        read_from_paths=None,
        chunksize=None,
        aggregate_files=None,
        **kwargs, ):
    from dask.dataframe.io.parquet.core \
        import tokenize, natural_sort_key, process_statistics, \
        set_index_columns, NONE_LABEL, DataFrameIOLayer, ParquetFunctionWrapper, HighLevelGraph, \
        new_dd_object, get_engine, stringify_path

    if isinstance(columns, str):
        df = read_parquet(
            path,
            columns=[columns],
            filters=filters,
            categories=categories,
            index=index,
            storage_options=storage_options,
            engine=engine,
            gather_statistics=gather_statistics,
            split_row_groups=split_row_groups,
            read_from_paths=read_from_paths,
            chunksize=chunksize,
            aggregate_files=aggregate_files,
        )
        return df[columns]

    if columns is not None:
        columns = list(columns)

    label = "read-parquet-"
    output_name = label + tokenize(
        path,
        columns,
        filters,
        categories,
        index,
        storage_options,
        engine,
        gather_statistics,
        split_row_groups,
        read_from_paths,
        chunksize,
        aggregate_files,
    )

    if isinstance(engine, str):
        engine = get_engine(engine)

    if hasattr(path, "name"):
        path = stringify_path(path)
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

    if chunksize or (
            split_row_groups and int(split_row_groups) > 1 and aggregate_files
    ):
        # Require `gather_statistics=True` if `chunksize` is used,
        # or if `split_row_groups>1` and we are aggregating files.
        if gather_statistics is False:
            raise ValueError("read_parquet options require gather_statistics=True")
        gather_statistics = True

    read_metadata_result = engine.read_metadata(
        fs,
        paths,
        categories=categories,
        index=index,
        gather_statistics=gather_statistics,
        filters=filters,
        split_row_groups=split_row_groups,
        read_from_paths=read_from_paths,
        chunksize=chunksize,
        aggregate_files=aggregate_files,
        **kwargs,
    )

    # In the future, we may want to give the engine the
    # option to return a dedicated element for `common_kwargs`.
    # However, to avoid breaking the API, we just embed this
    # data in the first element of `parts` for now.
    # The logic below is inteded to handle backward and forward
    # compatibility with a user-defined engine.
    meta, statistics, parts, index = read_metadata_result[:4]
    common_kwargs = {}
    aggregation_depth = False
    if len(parts):
        # For now, `common_kwargs` and `aggregation_depth`
        # may be stored in the first element of `parts`
        common_kwargs = parts[0].pop("common_kwargs", {})
        aggregation_depth = parts[0].pop("aggregation_depth", aggregation_depth)

    # Parse dataset statistics from metadata (if available)
    parts, divisions, index, index_in_columns = process_statistics(
        parts,
        statistics,
        filters,
        index,
        chunksize,
        split_row_groups,
        fs,
        aggregation_depth,
    )

    # Account for index and columns arguments.
    # Modify `meta` dataframe accordingly
    meta, index, columns = set_index_columns(
        meta, index, columns, index_in_columns, auto_index_allowed
    )
    if meta.index.name == NONE_LABEL:
        meta.index.name = None

    # Set the index that was previously treated as a column
    if index_in_columns:
        meta = meta.set_index(index)
        if meta.index.name == NONE_LABEL:
            meta.index.name = None

    if len(divisions) < 2:
        # empty dataframe - just use meta
        graph = {(output_name, 0): meta}
        divisions = (None, None)
    else:
        # Create Blockwise layer
        layer = DataFrameIOLayer(
            output_name,
            columns,
            parts,
            ParquetFunctionWrapper(
                engine,
                fs,
                meta,
                columns,
                index,
                kwargs,
                common_kwargs,
            ),
            label=label,
        )
        graph = HighLevelGraph({output_name: layer}, {output_name: set()})

    return new_dd_object(graph, output_name, meta, divisions)


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
