# -*- coding:utf-8 -*-
"""

"""
import pytest
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from hypernets.tabular import sklearn_ex as skex
from hypernets.tabular.column_selector import *
from hypernets.tabular.dataframe_mapper import DataFrameMapper
from hypernets.tabular.datasets import dsutils
from hypernets.utils import const


def get_df():
    X = pd.DataFrame(
        {
            "a": ['a', 'b', np.nan],
            "b": list(range(1, 4)),
            "c": np.arange(3, 6).astype("u1"),
            "d": np.arange(4.0, 7.0, dtype="float64"),
            "e": [True, False, True],
            "f": pd.Categorical(['c', 'd', np.nan]),
            "g": pd.date_range("20130101", periods=3),
            "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
            "i": pd.date_range("20130101", periods=3, tz="CET"),
            "j": pd.period_range("2013-01", periods=3, freq="M"),
            "k": pd.timedelta_range("1 day", periods=3),
            "l": [1, 10, 1000]
        }
    )
    y = [1, 1, 0]
    return X, y


class Test_Transformer():
    @classmethod
    def setup_class(cls):
        cls.bank_data = dsutils.load_bank()
        cls.movie_lens = dsutils.load_movielens()

    def test_func_transformer(self):
        dfm = DataFrameMapper(
            [(column_object_category_bool, [
                # SimpleImputer(strategy='constant'),
                skex.SafeSimpleImputer(strategy='constant'),
                skex.MultiLabelEncoder(),
            ]
              ),
             ],
            input_df=True,
            df_out=True,
            df_out_dtype_transforms=[
                (column_object, 'category')
            ]
        )
        X, y = get_df()
        x_new = dfm.fit_transform(X, y)
        assert x_new.dtypes.to_list() == [pd.CategoricalDtype(categories=[0, 1, 2], ordered=False),
                                          pd.CategoricalDtype(categories=[0, 1], ordered=False),
                                          pd.CategoricalDtype(categories=[0, 1, 2], ordered=False)]

    def test_pca(self):
        ct = make_column_transformer(
            (PCA(2), column_number_exclude_timedelta)
        )

        X, y = get_df()
        x_new = ct.fit_transform(X, y)
        assert x_new.shape == (3, 2)

        dfm = DataFrameMapper(
            [(column_number_exclude_timedelta, PCA(2)),
             # (column_object_category_bool, [SimpleImputer(strategy='constant'), OneHotEncoder()]),
             (column_object_category_bool, [skex.SafeSimpleImputer(strategy='constant'), OneHotEncoder()]),
             (column_number_exclude_timedelta, PolynomialFeatures(2)),
             ], input_df=True, df_out=True
        )
        x_new = dfm.fit_transform(X, y)
        # assert x_new.columns.to_list() == ['b_c_d_l_0', 'b_c_d_l_1', 'a_a', 'a_b', 'a_missing_value', 'e_False',
        #                                    'e_True', 'f_c', 'f_d', 'f_missing_value', '1', 'b', 'c', 'd', 'l',
        #                                    'b^2', 'b c', 'b d', 'b l', 'c^2', 'c d', 'c l', 'd^2', 'd l', 'l^2']
        # assert x_new.columns.to_list() == ['b_c_d_l_0', 'b_c_d_l_1', 'a_a', 'a_b', 'a_missing_value', 'e_False',
        #                                    'e_True', 'f_c', 'f_d', 'f_missing_value']
        assert {'b_c_d_l_0', 'b_c_d_l_1'}.issubset(set(x_new.columns.to_list()))

    def test_no_feature(self):
        df = get_df()[0]
        dfm = DataFrameMapper(
            [([], preprocessing.LabelEncoder())],
            input_df=True,
            df_out=True)

        with pytest.raises(ValueError):  # ValueError: No data output, maybe it's because your input feature is empty.
            dfm.fit_transform(df, None)

    def test_no_categorical_feature(self):
        df = get_df()[0][['b', 'd']]

        dfm = DataFrameMapper(
            [(column_object_category_bool, preprocessing.LabelEncoder())],
            input_df=True,
            df_out=True, default=None)

        x_new = dfm.fit_transform(df, None)

        assert 'b' in x_new
        assert 'd' in x_new

    def test_subsample(self):
        df = self.bank_data.copy()
        y = df.pop('y')
        X_train, X_test, y_train, y_test = skex.subsample(df, y, 100, 60, 'regression')
        assert X_train.shape == (60, 17)
        assert X_test.shape == (40, 17)
        assert y_train.shape == (60,)
        assert y_test.shape == (40,)

        X_train, X_test, y_train, y_test = skex.subsample(df, y, 100, 60, 'classification')
        assert X_train.shape == (60, 17)
        assert X_test.shape == (40, 17)
        assert y_train.shape == (60,)
        assert y_test.shape == (40,)

    def test_feature_selection(self):
        df = self.bank_data.copy()
        y = df.pop('y')
        reserved_cols = ['age', 'poutcome', 'id']
        fse = skex.FeatureSelectionTransformer('classification', 10000, 10000, 10, n_max_cols=8,
                                               reserved_cols=reserved_cols)
        fse.fit(df, y)
        assert len(fse.scores_.items()) == 10
        assert len(fse.columns_) == 11
        assert len(set(reserved_cols) - set(fse.columns_)) == 0

        x_t = fse.transform(df)
        assert x_t.columns.to_list() == fse.columns_

        df = dsutils.load_bank()
        y = df.pop('age')
        fse = skex.FeatureSelectionTransformer('regression', 10000, 10000, -1)
        fse.fit(df, y)
        assert len(fse.scores_.items()) == 17
        assert len(fse.columns_) == 10

    def test_feature_selection_by_importances(self):
        df = self.bank_data.copy()
        y = df.pop('y')
        fse = skex.FeatureImportancesSelectionTransformer('binary', strategy='number', number=5)
        fse.fit(df, y)
        assert fse.n_features_in_ == len(df.columns)
        assert len(fse.selected_features_) == 5

        x_t = fse.transform(df)
        assert x_t.columns.to_list() == fse.selected_features_

    def test_multi_label_encoder(self):
        df = pd.DataFrame({"A": [1, 2, 3, 4],
                           "B": ['a', 'a', 'a', 'b']})
        df_expect = pd.DataFrame({"A": [0, 1, 2, 3],
                                  "B": [0, 0, 0, 1]})
        ec = skex.MultiLabelEncoder(dtype=np.int64)
        df_t = ec.fit_transform(df.copy())
        assert np.where(df_expect.values == df_t.values, 0, 1).sum() == 0
        assert all(df_t.dtypes == pd.Series(dict(A=np.int64, B=np.int64)))

        ec = skex.MultiLabelEncoder(dtype=np.int32)
        df_t = ec.fit_transform(df.copy())
        assert np.where(df_expect.values == df_t.values, 0, 1).sum() == 0
        assert all(df_t.dtypes == pd.Series(dict(A=np.int32, B=np.int32)))

    def test_ordinal_encoder(self):
        df1 = pd.DataFrame({"A": [1, 2, 3, 4],
                            "B": ['a', 'a', 'a', 'b']})
        df2 = pd.DataFrame({"A": [1, 2, 3, 5],
                            "B": ['a', 'b', 'z', '0']})

        ec = skex.SafeOrdinalEncoder(dtype=np.int32)
        df = ec.fit_transform(df1)
        df_expect = pd.DataFrame({"A": [0, 1, 2, 3],
                                  "B": [0, 0, 0, 1]})
        # diff = (df - df_expect).values
        # assert np.count_nonzero(diff) == 0
        assert np.where(df_expect.values == df.values, 0, 1).sum() == 0

        df = ec.transform(df2)
        df_expect = pd.DataFrame({"A": [0, 1, 2, 4],
                                  "B": [0, 1, 2, 2]})
        assert np.where(df_expect.values == df.values, 0, 1).sum() == 0

        df = ec.inverse_transform(df_expect)
        df_expect = pd.DataFrame({"A": [1, 2, 3, -1],
                                  "B": ['a', 'b', None, None]})
        assert np.where(df_expect.values == df.values, 0, 1).sum() == 0

    def test_lgbm_leaves_encoder_binary(self):
        X = self.bank_data.copy()
        y = X.pop('y')
        cats = X.select_dtypes(['int', 'int64', ]).columns.to_list()
        continous = X.select_dtypes(['float', 'float64']).columns.to_list()
        n_estimators = 50
        t = skex.LgbmLeavesEncoder(cat_vars=cats, cont_vars=continous, task=const.TASK_BINARY,
                                   n_estimators=n_estimators)
        X = t.fit_transform(X[cats + continous].copy(), y)
        assert getattr(t.lgbm, 'n_estimators', 0) > 0
        assert len(X.columns) == len(cats) + len(continous) + t.lgbm.n_estimators

    def test_lgbm_leaves_encoder_multiclass(self):
        X = self.bank_data.copy()
        y = X.pop('age')
        cats = X.select_dtypes(['int', 'int64', ]).columns.to_list()
        continous = X.select_dtypes(['float', 'float64']).columns.to_list()
        t = skex.LgbmLeavesEncoder(cat_vars=cats, cont_vars=continous, task=const.TASK_MULTICLASS)
        X = t.fit_transform(X[cats + continous].copy(), y)
        assert getattr(t.lgbm, 'n_estimators', 0) > 0
        assert len(X.columns) == len(cats) + len(continous) + t.lgbm.n_classes_ * t.lgbm.n_estimators

    def test_lgbm_leaves_encoder_regression(self):
        X = self.bank_data.copy()
        y = X.pop('age').astype('float')
        cats = X.select_dtypes(['int', 'int64', ]).columns.to_list()
        continous = X.select_dtypes(['float', 'float64']).columns.to_list()
        n_estimators = 50
        t = skex.LgbmLeavesEncoder(cat_vars=cats, cont_vars=continous, task=const.TASK_REGRESSION,
                                   n_estimators=n_estimators)
        X = t.fit_transform(X[cats + continous].copy(), y)
        assert getattr(t.lgbm, 'n_estimators', 0) > 0
        assert len(X.columns) == len(cats) + len(continous) + t.lgbm.n_estimators

    def test_cat_encoder(self):
        X = self.bank_data.copy()
        y = X.pop('y')
        cats = X.select_dtypes(['int', 'int64', ]).columns.to_list()
        t = skex.CategorizeEncoder(columns=cats)
        Xt = t.fit_transform(X.copy(), y)
        assert len(t.new_columns) == len(cats)
        assert len(Xt.columns) == len(X.columns) + len(t.new_columns)

    def test_varlen_encoder_with_movie_lens(self):
        df = self.movie_lens.copy()
        df['genres_copy'] = df['genres']

        multi_encoder = skex.MultiVarLenFeatureEncoder([('genres', '|'), ('genres_copy', '|'), ])
        result_df = multi_encoder.fit_transform(df)

        assert multi_encoder.max_length_['genres'] > 0
        assert multi_encoder.max_length_['genres_copy'] > 0

        shape = np.array(result_df['genres'].tolist()).shape
        assert shape[1] == multi_encoder.max_length_['genres']

    def test_varlen_encoder_with_customized_data(self):
        from io import StringIO
        data = '''
            col_foo
            a|b|c|x
            a|b
            b|b|x
            '''.replace(' ', '')

        result = pd.DataFrame({'col_foo': [
            [1, 2, 3, 4],
            [1, 2, 0, 0],
            [2, 2, 4, 0]
        ]})
        df = pd.read_csv(StringIO(data))

        multi_encoder = skex.MultiVarLenFeatureEncoder([('col_foo', '|')])
        result_df = multi_encoder.fit_transform(df)
        print(result_df)

        assert all(result_df.values == result.values)

    def test_tfidf_encoder(self):
        df = self.movie_lens.copy()
        df['genres'] = df['genres'].apply(lambda s: s.replace('|', ' '))

        encoder = skex.TfidfEncoder(['title', 'genres'], flatten=False)
        Xt = encoder.fit_transform(df.copy())
        assert 'title' in Xt.columns.tolist()
        assert 'genres' in Xt.columns.tolist()
        assert isinstance(Xt['genres'][0], (list, np.ndarray))

        encoder = skex.TfidfEncoder(flatten=True)
        Xt = encoder.fit_transform(df[['title', 'genres']].copy())
        assert 'genres' not in Xt.columns.tolist()
        assert 'genres_tfidf_0' in Xt.columns.tolist()

        encoder = skex.TfidfEncoder(flatten=False)
        Xt = encoder.fit_transform(df[['title', 'genres']].copy().values)
        assert isinstance(Xt, np.ndarray)
        assert isinstance(Xt[0, 0], (list, np.ndarray))
        assert len(Xt[0, 1]) == 19
        assert Xt.shape == (df.shape[0], 2)

        encoder = skex.TfidfEncoder(flatten=False)
        Xt = encoder.fit_transform(df[['title']].copy().values)
        assert isinstance(Xt, np.ndarray)
        assert isinstance(Xt[0, 0], (list, np.ndarray))
        assert Xt.shape == (df.shape[0], 1)

        encoder = skex.TfidfEncoder(flatten=True)
        Xt = encoder.fit_transform(df[['genres']].copy().values)
        assert isinstance(Xt, np.ndarray)
        assert isinstance(Xt[0, 0], (int, float))
        assert Xt.shape == (df.shape[0], 19)

    def test_datetime_encoder(self):
        def is_holiday(x):
            holidays = {'0501', '0502', '0503'}
            return x.apply(lambda t: int(t.strftime('%m%d') in holidays))

        months = {'oct': 10, 'may': 5, 'apr': 4, 'jun': 6, 'feb': 2, 'aug': 8, 'jan': 1, 'jul': 7, 'nov': 11,
                  'sep': 9, 'mar': 3, 'dec': 12}

        df = dsutils.load_bank().sample(n=1000, random_state=9527)
        df['year'] = 2000
        df['month'] = df['month'].apply(lambda s: months[s])
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

        encoder = skex.DatetimeEncoder()
        X = encoder.fit_transform(df)
        columns = X.columns.to_list()
        assert 'date' not in columns
        assert all([c in columns for c in ['date_month', 'date_day']])
        assert all([c not in columns for c in ['date_hour', 'date_minute']])

        encoder = skex.DatetimeEncoder(include=skex.DatetimeEncoder.default_include + ['timestamp'],
                                       extra=[('holiday', is_holiday)], drop_constants=False)
        X = encoder.fit_transform(df)
        columns = X.columns.to_list()
        assert 'date' not in columns
        assert all([c in columns for c in ['date_holiday', 'date_timestamp']])

    def test_target_encoder(self):
        df = self.bank_data.head(1000)
        X_fit = df.head(300)
        y_fit = X_fit.pop('y')

        te = skex.TargetEncoder(smooth=2, seed=123)
        yt = skex.LabelEncoder().fit_transform(y_fit)
        Xt1 = te.fit_transform(X_fit['age'].to_frame(), yt)
        assert Xt1.shape == (len(X_fit),)

        mte = skex.MultiTargetEncoder(smooth=2, seed=123)
        Xtm = mte.fit_transform(X_fit, y_fit)
        assert Xtm.shape == X_fit.shape

        # compare fit_transform result
        assert (Xtm['age'].values == Xt1).all()

        df.pop('y')
        Xt1 = te.transform(df['age'].to_frame())
        assert Xt1.shape == (len(df),)

        Xtm = mte.transform(df)
        assert Xtm.shape == df.shape

        # compare transform result
        assert (Xtm['age'].values == Xt1).all()
