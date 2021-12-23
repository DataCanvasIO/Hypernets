import os.path as path

import pandas as pd

from hypernets.tabular import get_tool_box
from hypernets.tabular.datasets import dsutils
from . import if_dask_ready, is_dask_installed

if is_dask_installed:
    import dask.dataframe as dd
    from hypernets.tabular.dask_ex import DaskToolBox


@if_dask_ready
class TestDaskToolBox:
    def test_get_tool_box(self):
        tb = get_tool_box(dd.DataFrame)
        assert tb is DaskToolBox

        ddf = dd.from_pandas(pd.DataFrame(dict(
            x1=['a', 'b', 'c'],
            x2=[1, 2, 3]
        )), npartitions=1)
        tb = get_tool_box(ddf)
        assert tb is DaskToolBox

    def test_concat_df(self):
        df = pd.DataFrame(dict(
            x1=['a', 'b', 'c'],
            x2=[1, 2, 3]
        ))
        ddf = dd.from_pandas(df, npartitions=2)
        tb = get_tool_box(ddf)

        # DataFrame + DataFrame
        df1 = tb.concat_df([ddf, ddf], axis=0)
        assert isinstance(df1, dd.DataFrame)

        df1 = df1.compute()
        df2 = pd.concat([df, df], axis=0).reset_index(drop=True)
        assert (df1 == df2).all().all()

        # DataFrame + array
        df1 = tb.concat_df([ddf, ddf.to_dask_array(lengths=True)], axis=0)
        assert isinstance(df1, dd.DataFrame)

        df1 = df1.compute()
        df2 = pd.concat([df, df], axis=0).reset_index(drop=True)
        assert (df1 == df2).all().all()

    def test_load_data(self, ):
        data_dir = path.split(dsutils.__file__)[0]
        data_file = f'{data_dir}/blood.csv'

        df = DaskToolBox.load_data(data_file, reset_index=True)
        assert isinstance(df, dd.DataFrame)
