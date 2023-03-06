import glob
import os

import pytest

from hypernets.tabular.datasets import dsutils
from hypernets.tests import test_output_dir
from hypernets.utils import fs
from . import if_dask_ready, is_dask_installed, setup_dask

is_parquet_ready = False
if is_dask_installed:
    import dask.dataframe as dd
    import dask.array as da
    from hypernets.tabular.dask_ex import DaskToolBox

    try:
        p = DaskToolBox.parquet()
        is_parquet_ready = True
    except:
        pass


@pytest.mark.skipif(not is_parquet_ready, reason='ParquetPersistence is not installed')
@if_dask_ready
class TestDaskPersistence:
    @classmethod
    def setup_class(cls):
        setup_dask(cls)
        os.makedirs(f'{test_output_dir}/{cls.__name__}')
        fs.mkdirs(f'/{cls.__name__}', exist_ok=True)

    @classmethod
    def teardown_class(cls):
        fs.rm(f'/{cls.__name__}', recursive=True)

    @staticmethod
    def is_same_df(df1, df2):
        assert len(df1) == len(df2)
        assert df1.shape[1] == df2.shape[1]
        assert all(df1.columns == df2.columns)
        assert all(df1.dtypes == df2.dtypes)

        return True

    def test_dataframe(self):
        file_path = f'{test_output_dir}/{type(self).__name__}/test_df.parquet'
        os.makedirs(file_path, exist_ok=True)
        df = dsutils.load_bank_by_dask().repartition(npartitions=3)
        p.store(df, file_path)
        assert len(glob.glob(f'{file_path}/*.parquet')) == df.npartitions

        # read with pandas
        df_dd = dd.read_parquet(file_path)
        assert self.is_same_df(df, df_dd)

        # read with our utility
        df_read = p.load(file_path)
        assert self.is_same_df(df, df_read)

    #
    # def test_dataframe_int_columns(self):
    #     file_path = f'{test_output_dir}/{type(self).__name__}/test_df_ic.parquet'
    #     os.makedirs(file_path, exist_ok=True)
    #     df = dsutils.load_bank_by_dask().repartition(npartitions=3)
    #     # df.columns =list(map(str, range(len(df.columns))))
    #     df.columns = list(range(len(df.columns)))
    #     p.store(df, file_path)
    #     assert len(glob.glob(f'{file_path}/*.parquet')) == df.npartitions
    #
    #     # read with pandas
    #     df_dd = dd.read_parquet(file_path)
    #     assert self.is_same_df(df, df_dd)
    #
    #     # read with our utility
    #     df_read = p.load(file_path)
    #     assert self.is_same_df(df, df_read)

    def test_array(self):
        file_path = f'{test_output_dir}/{type(self).__name__}/test_ndarray.parquet'
        os.makedirs(file_path, exist_ok=True)
        df = dsutils.load_bank_by_dask().repartition(npartitions=3)
        p.store(DaskToolBox.df_to_array(df), file_path)
        assert len(glob.glob(f'{file_path}/*.parquet')) == df.npartitions

        values = p.load(file_path)
        assert isinstance(values, da.Array)
        assert values.shape[1] == df.shape[1]

        df = df.compute()
        df_read = dd.from_dask_array(values, columns=df.columns).compute()
        df_read.reset_index(drop=True, inplace=True)
        assert all(df_read['y'] == df['y'])

    def test_series(self):
        file_path = f'{test_output_dir}/{type(self).__name__}/test_series.parquet'
        os.makedirs(file_path, exist_ok=True)
        df = dsutils.load_bank_by_dask().repartition(npartitions=3)
        p.store(df['age'], file_path)
        assert len(glob.glob(f'{file_path}/*.parquet')) == df.npartitions

        s = p.load(file_path)
        assert isinstance(s, dd.Series)
        assert s.name == 'age'
        assert len(s) == len(df)
        s = s.compute()
        s.reset_index(drop=True, inplace=True)
        df = df.compute()
        assert all(s == df['age'])

    def test_dataframe_fs(self):
        file_path = f'/{type(self).__name__}/test_df_fs.parquet'
        fs.makedirs(file_path, exist_ok=True)
        df = dsutils.load_bank_by_dask().repartition(npartitions=3)
        p.store(df, file_path, filesystem=fs)
        assert len(fs.glob(f'{file_path}/*.parquet')) == df.npartitions

        # read it
        df_read = p.load(file_path, filesystem=fs)
        assert self.is_same_df(df, df_read)
