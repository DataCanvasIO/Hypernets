from . import if_cuml_ready, is_cuml_installed
from ..psudo_labeling_test import TestPseudoLabeling as _TestPseudoLabeling

if is_cuml_installed:
    import cudf


@if_cuml_ready
class TestCumlPseudoLabeling(_TestPseudoLabeling):

    @staticmethod
    def load_data():
        df = _TestPseudoLabeling.load_data()
        return cudf.from_pandas(df)

    @staticmethod
    def is_quantile_exact():
        return False
