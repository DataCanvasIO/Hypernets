from hypernets.utils import df_utils
from hypernets.tabular.datasets import dsutils
import numpy as np


def test_as_array():
    pd_df = dsutils.load_bank()
    pd_series = pd_df['id']

    assert isinstance(df_utils.as_array(pd_series), np.ndarray)
    assert isinstance(df_utils.as_array(pd_series.values), np.ndarray)
    assert isinstance(df_utils.as_array(pd_series.values.tolist()), np.ndarray)

    installed_cudf = False
    try:
        import cudf
        import cupy
        installed_cudf = True
    except Exception as e:
        pass

    if installed_cudf:
        import cudf

        cudf_series = cudf.from_pandas(pd_df)['id']
        assert isinstance(df_utils.as_array(cudf_series), np.ndarray)
        assert isinstance(df_utils.as_array(cudf_series.values), np.ndarray)
