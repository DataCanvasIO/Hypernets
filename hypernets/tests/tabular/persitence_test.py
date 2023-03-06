import os
from os import path

import numpy as np
import pandas as pd
import pytest

from hypernets.tabular.datasets import dsutils
from hypernets.tests import test_output_dir
from hypernets.utils import fs

try:
    from hypernets.tabular.persistence import ParquetPersistence

    p = ParquetPersistence()
    is_parquet_persitence_ready = True
except:
    is_parquet_persitence_ready = False


@pytest.mark.skipif(not is_parquet_persitence_ready, reason='ParquetPersistence is not installed')
class TestPersistence:
    @classmethod
    def setup_class(cls):
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
        df = dsutils.load_bank()
        p.store(df, file_path)
        assert path.exists(file_path)

        # read with pandas
        df_pd = pd.read_parquet(file_path)
        assert self.is_same_df(df, df_pd)

        # read with our utility
        df_read = p.load(file_path)
        assert self.is_same_df(df, df_read)

    def test_dataframe_int_columns(self):
        file_path = f'{test_output_dir}/{type(self).__name__}/test_df_ic.parquet'
        df = dsutils.load_bank()
        df.columns = range(len(df.columns))
        p.store(df, file_path)
        assert path.exists(file_path)

        # read with pandas
        df_pd = pd.read_parquet(file_path)
        assert self.is_same_df(df, df_pd)

        # read with our utility
        df_read = p.load(file_path)
        assert self.is_same_df(df, df_read)

    def test_ndarray(self):
        file_path = f'{test_output_dir}/{type(self).__name__}/test_ndarray.parquet'
        df = dsutils.load_bank()
        p.store(df.values, file_path)
        assert path.exists(file_path)

        values = p.load(file_path)
        assert isinstance(values, np.ndarray)
        assert values.shape == df.shape

        df_read = pd.DataFrame(values, columns=df.columns)
        assert all(df_read['y'] == df['y'])

    def test_series(self):
        file_path = f'{test_output_dir}/{type(self).__name__}/test_series.parquet'
        df = dsutils.load_bank()
        p.store(df['age'], file_path)
        assert path.exists(file_path)

        s = p.load(file_path)
        assert isinstance(s, pd.Series)
        assert s.name == 'age'
        assert len(s) == len(df)
        assert all(s == df['age'])

    def test_dataframe_fs(self):
        file_path = f'/{type(self).__name__}/test_df_fs.parquet'
        df = dsutils.load_bank()
        p.store(df, file_path, filesystem=fs)
        assert fs.exists(file_path)

        # read it
        df_read = p.load(file_path, filesystem=fs)
        assert self.is_same_df(df, df_read)
