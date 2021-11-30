from collections import Counter
from math import ceil

from hypernets.tabular import get_tool_box
from hypernets.tabular import sklearn_ex as skex
from hypernets.tabular.datasets import dsutils


class TestPseudoLabeling:
    @classmethod
    def setup_class(cls):
        cls.df = cls.load_data()

    @staticmethod
    def load_data():
        df = dsutils.load_bank()
        return skex.MultiLabelEncoder().fit_transform(df)

    def run_sample(self, X, y):
        tb = get_tool_box(X, y)
        model = tb.general_estimator(X, y)

        X_train, X_test, y_train, y_test = \
            tb.train_test_split(X, y, test_size=0.5, random_state=7)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)

        preds = model.predict(X_test)
        preds, = tb.to_local(preds)
        c0 = Counter(preds)
        print('original samples:', c0)

        options = dict(threshold=0.8, number=10, quantile=0.8)
        for strategy in ['threshold', 'number', 'quantile', ]:
            pl = tb.pseudo_labeling(strategy=strategy, **options)
            X_pseudo, y_pseudo = pl.select(X_test.copy(), model.classes_, proba.copy())

            y_pseudo, = tb.to_local(y_pseudo)

            # validate result data
            if len(y_pseudo) > 0:
                expected_y_pseudo = model.predict(X_pseudo)
                expected_y_pseudo, = tb.to_local(expected_y_pseudo)
                assert (expected_y_pseudo == y_pseudo).all()

            # validate sample numbers
            c = Counter(y_pseudo)
            if strategy == 'number':
                assert all([v <= options['number'] for k, v in c.items()])
            elif strategy == 'quantile':
                if self.is_quantile_exact():
                    expected_c = {k: ceil(c0[k] * (1 - options['quantile'])) for k, v in c0.items()}
                    assert c == expected_c

    @staticmethod
    def is_quantile_exact():
        return True

    def test_binary(self):
        X = self.df.copy()
        y = X.pop('y')
        self.run_sample(X, y)

    def test_multiclass(self):
        X = self.df.copy()
        y = X.pop('education')
        self.run_sample(X, y)
