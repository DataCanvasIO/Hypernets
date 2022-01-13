# -*- coding:utf-8 -*-
"""

"""
import os
import pickle
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from hypernets.tabular.datasets import dsutils
from . import if_cuml_ready, is_cuml_installed
from ... import test_output_dir

if is_cuml_installed:
    import cudf
    from hypernets.tabular.cuml_ex import CumlToolBox


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
        assert df1.dtypes.tolist() == df2.dtypes.tolist(), 'The same column dtypes were required.'

    if values:
        if not columns:
            df2.columns = df1.columns

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

    def fit_reload_transform(self, tf, *, df=None, target=None, column_selector=None, dtype=None, check_options=None):
        if df is None:
            df = self.bank_data.copy()
            target = 'y'

        if target is not None:
            y = df.pop(target)
            y_cf = cudf.from_pandas(y)
        else:
            y_cf = None

        if column_selector:
            columns = column_selector(df)
            df = df[columns]
        if dtype is not None:
            df = df.astype(dtype)
        cf = cudf.from_pandas(df)

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
        assert CumlToolBox.is_cuml_object(t)

        # convert to local transformer
        assert hasattr(tf_loaded, 'as_local')
        tf_local = tf_loaded.as_local()
        t2 = tf_local.transform(df.copy())
        assert isinstance(t2, (pd.DataFrame, np.ndarray))

        if check_options is None:
            check_options = {}
        check_dataframe(t, t2, **check_options)

    def test_standard_scaling(self):
        tf = CumlToolBox.transformers['StandardScaler']()
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_number)

    def test_maxabs_scaling(self):
        tf = CumlToolBox.transformers['MaxAbsScaler']()
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_number)

    def test_minmax_scaling(self):
        tf = CumlToolBox.transformers['MinMaxScaler']()
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_number)

    def test_truncated_svd(self):
        tf = CumlToolBox.transformers['TruncatedSVD'](n_components=3)
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_number,
                                  dtype='float64', check_options=dict(columns=False), )

    def test_robust_scaling(self):
        tf = CumlToolBox.transformers['RobustScaler']()
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_number)

    def test_impute_number_mean(self):
        tf = CumlToolBox.transformers['SimpleImputer'](strategy='mean')
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_number,
                                  dtype='float64', check_options=dict(columns=False), )

    def test_impute_number_median(self):
        tf = CumlToolBox.transformers['SimpleImputer'](strategy='median')
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_number,
                                  dtype='float64', check_options=dict(columns=False), )

    def test_impute_number_most_frequent(self):
        tf = CumlToolBox.transformers['SimpleImputer'](strategy='most_frequent')
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_number,
                                  dtype='float64', check_options=dict(columns=False), )

    def test_impute_number_constant(self):
        tf = CumlToolBox.transformers['SimpleImputer'](strategy='constant', fill_value=99.99)
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_number,
                                  dtype='float64', check_options=dict(columns=False), )

    def test_impute_object_constant(self):
        tf = CumlToolBox.transformers['ConstantImputer'](fill_value='<filled>')
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_object)

    def test_multi_label_encoder(self):
        tf = CumlToolBox.transformers['MultiLabelEncoder'](dtype=np.int32)
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_object)

    def test_onehot_encoder(self):
        tf = CumlToolBox.transformers['OneHotEncoder'](sparse=False)
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_object)

    def test_slim_target_encoder(self):
        tf = CumlToolBox.transformers['SlimTargetEncoder']()
        self.fit_reload_transform(tf, column_selector=lambda _: ['age', 'job', 'education'])

    def test_multi_target_encoder(self):
        tf = CumlToolBox.transformers['MultiTargetEncoder']()
        self.fit_reload_transform(tf, column_selector=CumlToolBox.column_selector.column_object)

    def test_onehot_svd_pipeline(self):
        ohe = CumlToolBox.transformers['OneHotEncoder'](sparse=False)
        svd = CumlToolBox.transformers['TruncatedSVD'](n_components=3)
        pipeline = CumlToolBox.transformers['Pipeline']([('onehot_encoder', ohe), ('svd', svd)])

        self.fit_reload_transform(pipeline, column_selector=CumlToolBox.column_selector.column_object)

    def test_tfidf_encoder(self):
        tf = CumlToolBox.transformers['TfidfEncoder']()
        self.fit_reload_transform(tf, df=self.movie_lens,
                                  column_selector=lambda _: ['title', 'genres'],
                                  check_options=dict(dtypes=False, values=False))

    def test_tfidf_encoder_flatten(self):
        tf = CumlToolBox.transformers['TfidfEncoder'](flatten=True)
        self.fit_reload_transform(tf, df=self.movie_lens,
                                  column_selector=lambda _: ['title', 'genres'],
                                  check_options=dict(dtypes=False))

    def test_tfidf_encoder_basic(self):
        df = self.movie_lens.copy()
        df['genres'] = df['genres'].apply(lambda s: s.replace('|', ' '))
        df = cudf.from_pandas(df)

        cls = CumlToolBox.transformers['TfidfEncoder']

        encoder = cls(['title', 'genres'], flatten=False)
        Xt = encoder.fit_transform(df.copy())
        assert isinstance(Xt, cudf.DataFrame)
        assert 'title' in Xt.columns.tolist()
        assert 'genres' in Xt.columns.tolist()
        assert isinstance(Xt['genres'][0], (list, np.ndarray))

        encoder = cls(flatten=True)
        Xt = encoder.fit_transform(df[['title', 'genres']].copy())
        assert isinstance(Xt, cudf.DataFrame)
        assert 'genres' not in Xt.columns.tolist()
        assert 'genres_tfidf_0' in Xt.columns.tolist()

    def test_datetime_encoder(self):
        df = self.movie_lens.copy()
        df['timestamp'] = df['timestamp'].apply(datetime.fromtimestamp)

        # fit with only datetime column
        tf = CumlToolBox.transformers['DatetimeEncoder']()
        self.fit_reload_transform(tf, df=df.copy(), column_selector=lambda _: ['timestamp', ],
                                  check_options=dict(dtypes=False))

        # fit with only datetime and object columns
        tf = CumlToolBox.transformers['DatetimeEncoder']()
        self.fit_reload_transform(tf, df=df.copy(), column_selector=lambda _: ['timestamp', 'genres'],
                                  check_options=dict(dtypes=False))

    def test_general_preprocessor(self):
        X_foo = cudf.from_pandas(self.bank_data.head())
        pp = CumlToolBox.general_preprocessor(X_foo)
        self.fit_reload_transform(pp, column_selector=CumlToolBox.column_selector.column_all,
                                  check_options=dict(dtypes=False))
