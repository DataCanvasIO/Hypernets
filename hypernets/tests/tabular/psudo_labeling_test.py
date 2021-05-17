from collections import Counter
from math import ceil

from lightgbm import LGBMClassifier

from hypernets.tabular import sklearn_ex as skex, dask_ex as dex
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.pseudo_labeling import sample_by_pseudo_labeling
from hypernets.tests.tabular.dask_transofromer_test import setup_dask


class Test_PseudoLabeling:

    def run_sample(self, X, y, with_dask=False):
        model = LGBMClassifier()

        if with_dask:
            setup_dask(None)
            X = dex.dd.from_pandas(X, npartitions=2)
            y = dex.dd.from_pandas(y, npartitions=2)
            model = dex.wrap_local_estimator(model)

        X_train, X_test, y_train, y_test = \
            dex.train_test_split(X, y, test_size=0.5, random_state=7)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)

        preds = model.predict(X_test)
        if dex.is_dask_object(preds):
            preds = dex.compute(preds)[0]
        c0 = Counter(preds)
        print('original samples:', c0)

        options = dict(threshold=0.8, number=10, quantile=0.8)
        for strategy in ['threshold', 'number', 'quantile', ]:
            X_pseudo, y_pseudo = \
                sample_by_pseudo_labeling(X_test.copy(), model.classes_, proba.copy(),
                                          strategy=strategy, **options)
            if dex.is_dask_object(y_pseudo):
                y_pseudo = dex.compute(y_pseudo)[0]

            # validate result data
            if len(y_pseudo) > 0:
                expected_y_pseudo = model.predict(X_pseudo)
                if dex.is_dask_object(expected_y_pseudo):
                    expected_y_pseudo = dex.compute(expected_y_pseudo)[0]
                assert (expected_y_pseudo == y_pseudo).all()

            # validate sample numbers
            c = Counter(y_pseudo)
            if strategy == 'number':
                assert all([v <= options['number'] for k, v in c.items()])
            elif strategy == 'quantile':
                if not with_dask:
                    expected_c = {k: ceil(c0[k] * (1 - options['quantile'])) for k, v in c0.items()}
                    assert c == expected_c

    def test_binary_sk(self):
        df = dsutils.load_bank().head(5000)
        X = skex.MultiLabelEncoder().fit_transform(df)
        y = X.pop('y')
        self.run_sample(X, y)

    def test_multiclass_sk(self):
        df = dsutils.load_bank().head(5000)
        X = skex.MultiLabelEncoder().fit_transform(df)
        y = X.pop('education')
        self.run_sample(X, y)

    def test_binary_dask(self):
        df = dsutils.load_bank().head(5000)
        X = skex.MultiLabelEncoder().fit_transform(df)
        y = X.pop('y')
        self.run_sample(X, y, with_dask=True)

    def test_multiclass_dask(self):
        df = dsutils.load_bank().head(5000)
        X = skex.MultiLabelEncoder().fit_transform(df)
        y = X.pop('education')
        self.run_sample(X, y, with_dask=True)


if __name__ == '__main__':
    pass
    # t = Foo_PseudoLabeling()
    # t.test_multiclass_sk()
    # t.test_binary_sk()
    # t.test_binary_dask()
    # t.test_multiclass_dask()
