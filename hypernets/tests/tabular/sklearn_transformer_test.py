# -*- coding:utf-8 -*-
"""

"""
import pytest
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from hypernets.tabular.column_selector import *
from hypernets.tabular.dataframe_mapper import DataFrameMapper
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.sklearn_ex import MultiLabelEncoder, FeatureSelectionTransformer, SafeOrdinalEncoder, subsample


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

    def test_func_transformer(self):
        dfm = DataFrameMapper(
            [(column_object_category_bool, [
                SimpleImputer(strategy='constant'),
                MultiLabelEncoder(),
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
             (column_object_category_bool, [SimpleImputer(strategy='constant'), OneHotEncoder()]),
             (column_number_exclude_timedelta, PolynomialFeatures(2)),
             ], input_df=True, df_out=True
        )
        x_new = dfm.fit_transform(X, y)
        assert x_new.columns.to_list() == ['b_c_d_l_0', 'b_c_d_l_1', 'a_a', 'a_b', 'a_missing_value', 'e_False',
                                           'e_True', 'f_c', 'f_d', 'f_missing_value', '1', 'b', 'c', 'd', 'l',
                                           'b^2', 'b c', 'b d', 'b l', 'c^2', 'c d', 'c l', 'd^2', 'd l', 'l^2']

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
        df = dsutils.load_bank()
        y = df.pop('y')
        X_train, X_test, y_train, y_test = subsample(df, y, 100, 60, 'regression')
        assert X_train.shape == (60, 17)
        assert X_test.shape == (40, 17)
        assert y_train.shape == (60,)
        assert y_test.shape == (40,)

        X_train, X_test, y_train, y_test = subsample(df, y, 100, 60, 'classification')
        assert X_train.shape == (60, 17)
        assert X_test.shape == (40, 17)
        assert y_train.shape == (60,)
        assert y_test.shape == (40,)

    def test_feature_selection(self):
        df = dsutils.load_bank()
        y = df.pop('y')
        reserved_cols = ['age', 'poutcome', 'id']
        fse = FeatureSelectionTransformer('classification', 10000, 10000, 10, n_max_cols=8, reserved_cols=reserved_cols)
        fse.fit(df, y)
        assert len(fse.scores_.items()) == 10
        assert len(fse.columns_) == 11
        assert len(set(reserved_cols) - set(fse.columns_)) == 0

        x_t = fse.transform(df)
        assert x_t.columns.to_list() == fse.columns_

        df = dsutils.load_bank()
        y = df.pop('age')
        fse = FeatureSelectionTransformer('regression', 10000, 10000, -1)
        fse.fit(df, y)
        assert len(fse.scores_.items()) == 17
        assert len(fse.columns_) == 10

    def test_ordinal_encoder(self):
        df1 = pd.DataFrame({"A": [1, 2, 3, 4],
                            "B": ['a', 'a', 'a', 'b']})
        df2 = pd.DataFrame({"A": [1, 2, 3, 5],
                            "B": ['a', 'b', 'z', '0']})

        ec = SafeOrdinalEncoder(dtype=np.int32)
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
