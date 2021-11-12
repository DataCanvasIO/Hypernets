import pandas as pd

from hypernets.tabular.datasets import dsutils
from hypernets.tabular.persistence import to_parquet, read_parquet
from hypernets.tests.tabular.dask_transofromer_test import setup_dask


class Test_Persistence:
    @classmethod
    def setup_class(cls):
        setup_dask(cls)

    def test_with_filesystem_dask(self):
        from hypernets.utils import fs

        target_file = 'tmp.parquet'
        fs.mkdirs(target_file, exist_ok=True)

        df = dsutils.load_bank_by_dask().repartition(npartitions=2)
        files = to_parquet(df, target_file, filesystem=fs)
        print('files:', files)
        assert len(files) == df.npartitions
        assert all(map(fs.exists, files))

        # reload with dask
        df_dd = read_parquet(target_file, delayed=True, filesystem=fs)
        df_dd = df_dd.compute()

        # reload with pandas
        dfs = [read_parquet(f, delayed=False, filesystem=fs) for f in files]
        df_pd = pd.concat(dfs, ignore_index=True)
        assert df_dd.shape == df_pd.shape

        fs.rm(target_file, recursive=True)
