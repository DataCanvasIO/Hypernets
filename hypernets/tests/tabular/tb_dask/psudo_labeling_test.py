from hypernets.tests.tabular.dask_transofromer_test import setup_dask
from . import if_dask_ready, is_dask_installed
from ..psudo_labeling_test import TestPseudoLabeling

if is_dask_installed:
    import dask.dataframe as dd


@if_dask_ready
class TestDaskPseudoLabeling(TestPseudoLabeling):
    @staticmethod
    def load_data():
        setup_dask(None)
        df = TestPseudoLabeling.load_data()
        return dd.from_pandas(df, npartitions=2)

    @staticmethod
    def is_quantile_exact():
        return False
