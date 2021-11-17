# -*- coding:utf-8 -*-
"""

"""
import os
import pickle
import shutil

import pandas as pd
import numpy as np
from hypernets.tabular.datasets import dsutils
from . import if_cuml_ready
from ... import test_output_dir


def check_dataframe(df1, df2, *, shape=True, columns=True, dtypes=True, values=True, delta=1e-5):
    from hypernets.tabular import get_tool_box

    if not isinstance(df1, pd.DataFrame):
        df1, = get_tool_box(df1).to_local(df1)
        df1 = pd.DataFrame(df1)
    if not isinstance(df2, pd.DataFrame):
        df2, = get_tool_box(df2).to_local(df2)
        df2 = pd.DataFrame(df2)

    if shape:
        assert df1.shape == df2.shape, 'The same dataframe shape is required.'

    if columns:
        assert all(df1.columns == df2.columns), 'The same column names were required.'

    if dtypes:
        assert all(df1.dtypes == df2.dtypes), 'The same column dtypes were required.'

    if values:
        float_cols = df1.select_dtypes(['float32', 'float64']).columns.tolist()
        if float_cols:
            df1_float = df1[float_cols]
            df2_float = df2[float_cols]
            value_diff = (df1_float - df2_float).abs().max().max()
            assert value_diff < delta

            df1_nofloat = df1[[c for c in df1.columns.tolist() if c not in float_cols]]
            df2_nofloat = df2[[c for c in df2.columns.tolist() if c not in float_cols]]
        else:
            df1_nofloat = df1
            df2_nofloat = df2

        if df1_nofloat.shape[1] > 0:
            assert (df1_nofloat == df2_nofloat).all().all(), 'all value should be equal.'

    return True


@if_cuml_ready
class TestCumlTransformer:
    import cudf
    from hypernets.tabular.cuml_ex import CumlToolBox

    work_dir = f'{test_output_dir}/Test_CumlTransformer'

    @classmethod
    def setup_class(cls):
        from sklearn.preprocessing import LabelEncoder
        df = dsutils.load_bank()
        df['y'] = LabelEncoder().fit_transform(df['y'])
        cls.bank_data = df
        cls.movie_lens = dsutils.load_movielens()

        os.makedirs(cls.work_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def fit_reload_transform(self, tf, *, column_selector=None, dtype=None, check_options=None):
        tb = self.CumlToolBox

        df = self.bank_data.copy()
        y = df.pop('y')
        if column_selector:
            columns = column_selector(df)
            df = df[columns]
        if dtype is not None:
            df = df.astype(dtype)

        cf = self.cudf.from_pandas(df)
        y_cf = self.cudf.from_pandas(y)

        tf.fit_transform(cf.copy(), y_cf)
        file_path = f'{self.work_dir}/fitted_{type(tf).__name__}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(tf, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(file_path, 'rb') as f:
            tf_loaded = pickle.load(f)
            assert type(tf_loaded) is type(tf)

        # transform cudf.DataFrame
        t = tf_loaded.transform(cf)
        assert t is not None
        assert tb.is_cuml_object(t)

        # convert to local transformer
        assert hasattr(tf_loaded, 'as_local')
        tf_local = tf_loaded.as_local()
        t2 = tf_local.transform(df.copy())
        assert isinstance(t2, (pd.DataFrame, np.ndarray))

        if check_options is None:
            check_options = {}
        check_dataframe(t, t2, **check_options)

    def test_standard_scaling(self):
        tf = self.CumlToolBox.transformers['StandardScaler']()
        self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_number)

    def test_maxabs_scaling(self):
        tf = self.CumlToolBox.transformers['MaxAbsScaler']()
        self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_number)

    def test_minmax_scaling(self):
        tf = self.CumlToolBox.transformers['MinMaxScaler']()
        self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_number)

    def test_truncated_svd(self):
        tf = self.CumlToolBox.transformers['TruncatedSVD'](n_components=3)
        self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_number,
                                  dtype='float64', check_options=dict(columns=False), )

    def test_robust_scaling(self):
        tf = self.CumlToolBox.transformers['RobustScaler']()
        self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_number)

    def test_impute_number_mean(self):
        tf = self.CumlToolBox.transformers['SimpleImputer'](strategy='mean')
        self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_number)

    def test_impute_number_median(self):
        tf = self.CumlToolBox.transformers['SimpleImputer'](strategy='median')
        self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_number)

    def test_impute_number_most_frequent(self):
        tf = self.CumlToolBox.transformers['SimpleImputer'](strategy='most_frequent')
        self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_number)

    def test_impute_number_constant(self):
        tf = self.CumlToolBox.transformers['SimpleImputer'](strategy='constant', fill_value=99.99)
        self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_number)

    def test_impute_object_constant(self):
        tf = self.CumlToolBox.transformers['ConstantImputer'](fill_value='<filled>')
        self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_object)

    def test_multi_labelencoder(self):
        tf = self.CumlToolBox.transformers['MultiLabelEncoder'](dtype=np.int32)
        self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_object)

    def test_onehot_encoder(self):
        tf = self.CumlToolBox.transformers['OneHotEncoder'](sparse=False)
        self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_object)

    def test_onehot_svd_pipeline(self):
        ohe = self.CumlToolBox.transformers['OneHotEncoder'](sparse=False)
        svd = self.CumlToolBox.transformers['TruncatedSVD'](n_components=3)
        pipeline = self.CumlToolBox.transformers['Pipeline']([('onehot_encoder', ohe), ('svd', svd)])

        self.fit_reload_transform(pipeline, column_selector=self.CumlToolBox.column_selector.column_object)

    #
    # def test_target_lencoder(self):
    #     tf = self.CumlToolBox.transformers['TargetEncoder']()
    #     self.fit_reload_transform(tf, column_selector=self.CumlToolBox.column_selector.column_object)

    def test_general_preprocessor(self):
        X_foo = self.cudf.from_pandas(self.bank_data.head())
        pp = self.CumlToolBox.general_preprocessor(X_foo)
        self.fit_reload_transform(pp, column_selector=self.CumlToolBox.column_selector.column_all,
                                  check_options=dict(dtypes=False))
