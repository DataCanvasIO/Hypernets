import numpy as np

from hypernets.tabular import sklearn_ex as skex
from hypernets.tabular.datasets import dsutils
from hypernets.utils import const


class TestCustomizedTransformers:
    @classmethod
    def setup_class(cls):
        cls.bank_data = dsutils.load_bank()
        cls.movie_lens = dsutils.load_movielens()

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

    def test_varlens_encoder(self):
        df = self.movie_lens.copy()
        df['genres_copy'] = df['genres']

        multi_encoder = skex.MultiVarLenFeatureEncoder([('genres', '|'), ('genres_copy', '|'), ])
        result_df = multi_encoder.fit_transform(df)

        assert multi_encoder._encoders['genres'].max_element_length > 0
        assert multi_encoder._encoders['genres_copy'].max_element_length > 0

        shape = np.array(result_df['genres'].tolist()).shape
        assert shape[1] == multi_encoder._encoders['genres'].max_element_length


if __name__ == '__main__':
    TestCustomizedTransformers.setup_class()
    t = TestCustomizedTransformers()

    t.test_varlens_encoder()
