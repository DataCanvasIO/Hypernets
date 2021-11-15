import os
from os import path

import pandas as pd

from hypernets.tabular.datasets import dsutils
from hypernets.tabular.persistence import to_parquet, read_parquet
from hypernets.tests import test_output_dir
from hypernets.tests.tabular.dask_transofromer_test import setup_dask
from hypernets.utils import fs


class Test_Persistence:
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

    def test_pandas(self):
        file_path = f'{test_output_dir}/{type(self).__name__}/test_pandas.parquet'
        df = dsutils.load_bank()
        to_parquet(df, file_path)
        assert path.exists(file_path)

        # read with pandas
        df_pd = pd.read_parquet(file_path)
        assert self.is_same_df(df, df_pd)

        # read with our utility
        df_read = read_parquet(file_path)
        assert self.is_same_df(df, df_read)

    def test_filesystem_pandas(self):
        file_path = f'/{type(self).__name__}/test_pandas_fs.parquet'
        df = dsutils.load_bank()
        to_parquet(df, file_path, filesystem=fs)
        assert fs.exists(file_path)

        # read it
        df_read = read_parquet(file_path, filesystem=fs)
        assert self.is_same_df(df, df_read)

    def test_dask(self):
        target_file = f'{test_output_dir}{type(self).__name__}/test_with_dask.parquet'
        os.makedirs(target_file, exist_ok=True)

        df = dsutils.load_bank_by_dask().repartition(npartitions=2)
        files = to_parquet(df, target_file)
        print('files:', files)
        assert len(files) == df.npartitions
        assert all(map(path.exists, files))

        # reload with dask
        df_dd = read_parquet(target_file, delayed=True)
        df_dd = df_dd.compute()
        assert self.is_same_df(df, df_dd)

        # reload with pandas
        dfs = [read_parquet(f, delayed=False) for f in files]
        df_pd = pd.concat(dfs, ignore_index=True)
        assert self.is_same_df(df, df_pd)

    def test_filesystem_dask(self):
        target_file = 'test_filesystem_dask.parquet'
        fs.mkdirs(target_file, exist_ok=True)

        df = dsutils.load_bank_by_dask().repartition(npartitions=2)
        files = to_parquet(df, target_file, filesystem=fs)
        print('files:', files)
        assert len(files) == df.npartitions
        assert all(map(fs.exists, files))

        # reload with dask
        df_dd = read_parquet(target_file, delayed=True, filesystem=fs)
        df_dd = df_dd.compute()
        assert self.is_same_df(df, df_dd)

        # reload with pandas
        dfs = [read_parquet(f, delayed=False, filesystem=fs) for f in files]
        df_pd = pd.concat(dfs, ignore_index=True)
        assert self.is_same_df(df, df_pd)
