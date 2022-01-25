import numpy as np
import pandas as pd

from hypernets.tabular.datasets import dsutils
from hypernets.utils import const
from . import if_dask_ready, is_dask_installed, setup_dask

if is_dask_installed:
    import dask.dataframe as dd
    import dask
    from hypernets.tabular import dask_ex as dex


@if_dask_ready
class Test_DaskCustomizedTransformer:
    @classmethod
    def setup_class(cls):
        setup_dask(cls)

        cls.bank_data = dd.from_pandas(dsutils.load_bank(), npartitions=2)
        cls.movie_lens = dd.from_pandas(dsutils.load_movielens(), npartitions=2)

    def test_lgbm_leaves_encoder_binary(self):
        X = self.bank_data.copy()
        y = X.pop('y')
        cats = X.select_dtypes(['int', 'int64', ]).columns.to_list()
        continous = X.select_dtypes(['float', 'float64']).columns.to_list()
        X = X[cats + continous]
        n_estimators = 50
        X_shape = dask.compute(X.shape)[0]

        # with fit_transform
        t = dex.LgbmLeavesEncoder(cat_vars=cats, cont_vars=continous, task=const.TASK_BINARY,
                                  n_estimators=n_estimators)
        Xt = t.fit_transform(X.copy(), y)

        # check attributes
        attributes = {'lgbm', 'cat_vars', 'cont_vars', 'new_columns', 'task', 'lgbm_params'}
        assert len(attributes - set(dir(t))) == 0
        assert all([hasattr(t, a) for a in attributes])
        assert all([getattr(t, a, None) is not None for a in attributes])

        # check result
        Xt = Xt.compute()
        assert getattr(t.lgbm, 'n_estimators', 0) > 0
        assert len(Xt.columns) == len(cats) + len(continous) + t.lgbm.n_estimators
        assert X_shape[0] == Xt.shape[0]

        # with fit + transform
        t2 = dex.LgbmLeavesEncoder(cat_vars=cats, cont_vars=continous, task=const.TASK_BINARY,
                                   n_estimators=n_estimators)
        Xt = t2.fit(X.copy(), y).transform(X.copy())
        Xt = Xt.compute()
        assert getattr(t2.lgbm, 'n_estimators', 0) > 0
        assert len(Xt.columns) == len(cats) + len(continous) + t2.lgbm.n_estimators
        assert X_shape[0] == Xt.shape[0]

    def test_cat_encoder(self):
        X = self.bank_data.copy()
        y = X.pop('y')
        cats = X.select_dtypes(['int', 'int64', ]).columns.to_list()
        t = dex.CategorizeEncoder(columns=cats)
        Xt = t.fit_transform(X.copy(), y)
        assert isinstance(Xt, dd.DataFrame)

        Xt = Xt.compute()
        assert len(t.new_columns) == len(cats)
        assert len(Xt.columns) == len(X.columns) + len(t.new_columns)

    def run_bins_discretizer(self, strategy, dtype):
        X = self.bank_data.copy()

        encoder = dex.MultiKBinsDiscretizer(['age', 'id'], strategy=strategy, dtype=dtype)
        X = encoder.fit_transform(X)
        assert set(encoder.bin_edges_.keys()) == {'age', 'id'}

        assert isinstance(X, dd.DataFrame)
        X = X.compute()

        for col, new_name, c_bins in encoder.new_columns:
            assert X[new_name].nunique() <= c_bins

            col_values = X[new_name].values
            assert col_values.min() == 0 and col_values.max() == c_bins - 1

            if dtype is not None and dtype.startswith('float'):
                assert str(X[new_name].dtype).startswith('float')
            else:
                assert str(X[new_name].dtype).startswith('int')

    def test_bins_discretizer_uniform_float64(self):
        self.run_bins_discretizer('uniform', 'float64')

    def test_bins_discretizer_quantile_float64(self):
        self.run_bins_discretizer('quantile', 'float64')

    def test_bins_discretizer_uniform_int(self):
        self.run_bins_discretizer('uniform', 'int')

    def test_bins_discretizer_quantile_int(self):
        self.run_bins_discretizer('uniform', None)

    def test_varlen_encoder_with_customized_data(self):
        from io import StringIO
        data = '''
        col_foo
        a|b|c|x
        a|b
        b|b|x
        '''.replace(' ', '')

        df = pd.read_csv(StringIO(data))
        result = pd.DataFrame({'col_foo': [
            [1, 2, 3, 4],
            [1, 2, 0, 0],
            [2, 2, 4, 0]
        ]})

        multi_encoder = dex.MultiVarLenFeatureEncoder([('col_foo', '|')])
        result_df = multi_encoder.fit_transform(df.copy())
        print(result_df)
        assert all(result_df.values == result.values)

        ddf = dd.from_pandas(df.copy(), npartitions=1)
        d_multi_encoder = dex.MultiVarLenFeatureEncoder([('col_foo', '|')])
        d_result_df = d_multi_encoder.fit_transform(ddf)
        assert isinstance(d_result_df, dd.DataFrame)
        d_result_df = d_result_df.compute()

        print(d_result_df)
        assert all(d_result_df.values == result.values)

    def test_dataframe_wrapper(self):
        X = self.bank_data.copy()

        cats = X.select_dtypes(['object', ]).columns.to_list()
        continous = X.select_dtypes(['float', 'float64', 'int', 'int64']).columns.to_list()
        transformers = [('cats',
                         dex.SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=''),
                         cats
                         ),
                        ('conts',
                         dex.SimpleImputer(missing_values=np.nan, strategy='mean'),
                         continous
                         )]
        ct = dex.ColumnTransformer(transformers=transformers)
        dfw = dex.DataFrameWrapper(ct, cats + continous)
        X = dfw.fit_transform(X[cats + continous])

        assert isinstance(X, dd.DataFrame)
        print(X.dtypes)


if __name__ == '__main__':
    Test_DaskCustomizedTransformer.setup_class()
    c = Test_DaskCustomizedTransformer()
    c.test_cat_encoder()
