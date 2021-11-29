from hypernets.tests.tabular.dask_transofromer_test import setup_dask
from . import if_dask_ready, is_dask_installed
from ..psudo_labeling_test import TestPseudoLabeling

if is_dask_installed:
    import dask.dataframe as dd


@if_dask_ready
class TestDaskPseudoLabeling(TestPseudoLabeling):
    @classmethod
    def setup_class(cls):
        TestPseudoLabeling.setup_class()
        cls.df = dd.from_pandas(TestPseudoLabeling.df, npartitions=2)

        setup_dask(cls)

    @staticmethod
    def is_quantile_exact():
        return False
