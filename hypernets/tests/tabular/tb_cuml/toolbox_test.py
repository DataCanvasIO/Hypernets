# -*- coding:utf-8 -*-
"""

"""
import os.path as path

import pytest

from hypernets.tabular import get_tool_box
from hypernets.tabular.datasets import dsutils
from . import if_cuml_ready, is_cuml_installed

if is_cuml_installed:
    import cudf
    from hypernets.tabular.cuml_ex import CumlToolBox


@if_cuml_ready
class TestCumlToolBox:

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

    def test_detect_estimator_cuml_rf(self):
        tb = get_tool_box(cudf.DataFrame)
        detector = tb.estimator_detector('cuml.RandomForestClassifier', 'binary')
        r = detector()
        assert r == {'installed', 'initialized', 'fitted', 'fitted_with_cuml'}

    def test_detect_estimator_lightgbm(self):
        tb = get_tool_box(cudf.DataFrame)
        detector = tb.estimator_detector('lightgbm.LGBMClassifier', 'binary',
                                         init_kwargs={'device': 'GPU'}, )
        r = detector()
        assert r == {'installed', 'initialized', 'fitted'}  # lightgbm dose not support cudf.DataFrame

    def test_detect_estimator_xgboost(self):
        pytest.importorskip('xgboost')

        tb = get_tool_box(cudf.DataFrame)
        detector = tb.estimator_detector('xgboost.XGBClassifier', 'binary',
                                         init_kwargs={'tree_method': 'gpu_hist', 'use_label_encoder': False}, )
        r = detector()
        assert r == {'installed', 'initialized', 'fitted', 'fitted_with_cuml'}

    def test_concat_df(self):
        df = cudf.DataFrame(dict(
            x1=['a', 'b', 'c'],
            x2=[1, 2, 3],
            x3=[4., 5, 6],
        ))
        tb = get_tool_box(cudf.DataFrame)

        # DataFrame + DataFrame
        df1 = tb.concat_df([df, df], axis=0)
        df2 = cudf.concat([df, df], axis=0)
        assert (df1 == df2).all().all()

        # DataFrame + ndarray
        df_num = df[['x2', 'x3']]
        df1 = tb.concat_df([df_num, df_num.values], axis=0)
        df2 = cudf.concat([df_num, df_num], axis=0)
        assert isinstance(df1, cudf.DataFrame)
        assert (df1 == df2).all().all()

        # Series + ndarray
        s = df['x2']
        df1 = tb.concat_df([s, s.values], axis=0)
        df2 = cudf.concat([s, s], axis=0)
        assert isinstance(df1, cudf.Series)
        assert (df1 == df2).all()

    def test_load_data(self, ):
        data_dir = path.split(dsutils.__file__)[0]
        data_file = f'{data_dir}/blood.csv'

        df = CumlToolBox.load_data(data_file, reset_index=True)
        assert isinstance(df, cudf.DataFrame)
