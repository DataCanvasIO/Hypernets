# -*- coding:utf-8 -*-
"""

"""

from hypernets.tabular import get_tool_box
from hypernets.tabular.datasets import dsutils
from . import if_cuml_ready, is_cuml_installed

if is_cuml_installed:
    import cudf
    from hypernets.tabular.cuml_ex import CumlToolBox


@if_cuml_ready
class TestCumlTransformer:

    @classmethod
    def setup_class(cls):
        from sklearn.preprocessing import LabelEncoder
        df = dsutils.load_bank()
        df['y'] = LabelEncoder().fit_transform(df['y'])
        df['education'] = LabelEncoder().fit_transform(df['education'])
        cf = cudf.from_pandas(df)

        cls.df = df
        cls.cf = cf

    def test_get_toolbox(self):
        tb = get_tool_box(self.cf)
        assert tb is CumlToolBox

    def test_general_preprocessor(self):
        X = self.cf.copy()
        y = X.pop('y')
        preprocessor = CumlToolBox.general_preprocessor(self.cf)
        Xt = preprocessor.fit_transform(X, y)
        assert CumlToolBox.is_cuml_object(Xt)

        # dtypes
        dtypes = set(map(str, Xt.dtypes.to_dict().values()))
        assert dtypes.issubset({'float64', 'int64', 'uint8'})

    def test_general_estimator(self):
        X = self.cf.copy()
        y = X.pop('y')
        preprocessor = CumlToolBox.general_preprocessor(self.cf)
        Xt = preprocessor.fit_transform(X, y)

        for s in [None, 'xgb', 'rf', 'gbm']:
            est = CumlToolBox.general_estimator(Xt, y, estimator=s)
            est.fit(Xt, y)
            assert len(est.classes_) == 2

            pred = est.predict(Xt)
            assert CumlToolBox.is_cuml_object(pred)

            proba = est.predict_proba(Xt)
            assert CumlToolBox.is_cuml_object(proba)
