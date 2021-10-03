from collections import Counter
from math import ceil

import dask
import dask.dataframe as dd
from lightgbm import LGBMClassifier

from hypernets.tabular import get_tool_box
from hypernets.tabular import sklearn_ex as skex
from hypernets.tabular.datasets import dsutils
from hypernets.tests.tabular.dask_transofromer_test import setup_dask


class Test_PseudoLabeling:

    def run_sample(self, X, y, with_dask=False):
        model = LGBMClassifier()

        if with_dask:
            setup_dask(None)
            X = dd.from_pandas(X, npartitions=2)
            y = dd.from_pandas(y, npartitions=2)
            tb = get_tool_box(X, y)
            model = tb.wrap_local_estimator(model)
        else:
            tb = get_tool_box(X, y)

        X_train, X_test, y_train, y_test = \
            tb.train_test_split(X, y, test_size=0.5, random_state=7)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)

        preds = model.predict(X_test)
        if isinstance(X, dd.DataFrame):
            preds = dask.compute(preds)[0]
        c0 = Counter(preds)
        print('original samples:', c0)

        options = dict(threshold=0.8, number=10, quantile=0.8)
        for strategy in ['threshold', 'number', 'quantile', ]:
            X_pseudo, y_pseudo = \
                tb.pseudo_labeling(strategy=strategy, **options).select(X_test.copy(), model.classes_, proba.copy())

            if isinstance(X, dd.DataFrame):
                y_pseudo = dask.compute(y_pseudo)[0]

            # validate result data
            if len(y_pseudo) > 0:
                expected_y_pseudo = model.predict(X_pseudo)
                if isinstance(X, dd.DataFrame):
                    expected_y_pseudo = dask.compute(expected_y_pseudo)[0]
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
        df = dsutils.load_bank()
        X = skex.MultiLabelEncoder().fit_transform(df)
        y = X.pop('y')
        self.run_sample(X, y)

    def test_multiclass_sk(self):
        df = dsutils.load_bank()
        X = skex.MultiLabelEncoder().fit_transform(df)
        y = X.pop('education')
        self.run_sample(X, y)

    def test_binary_dask(self):
        df = dsutils.load_bank()
        X = skex.MultiLabelEncoder().fit_transform(df)
        y = X.pop('y')
        self.run_sample(X, y, with_dask=True)

    def test_multiclass_dask(self):
        df = dsutils.load_bank()
        X = skex.MultiLabelEncoder().fit_transform(df)
        y = X.pop('education')
        self.run_sample(X, y, with_dask=True)


if __name__ == '__main__':
    pass
