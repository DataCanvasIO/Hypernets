from . import if_cuml_ready, is_cuml_installed
from ..psudo_labeling_test import TestPseudoLabeling

if is_cuml_installed:
    import cudf


@if_cuml_ready
class TestCumlPseudoLabeling(TestPseudoLabeling):
    @classmethod
    def setup_class(cls):
        TestPseudoLabeling.setup_class()
        cls.df = cudf.from_pandas(TestPseudoLabeling.df)

    @staticmethod
    def is_quantile_exact():
        return False
