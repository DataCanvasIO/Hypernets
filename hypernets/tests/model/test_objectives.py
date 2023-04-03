import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from hypernets.model.objectives import NumOfFeatures


class TestNumOfFeatures:

    def create_mock_dataset(self):
        X = np.random.random((10000, 4))
        df = pd.DataFrame(data=X, columns= [str("c_%s" % i) for i in range(4)])
        y = np.random.random(10000)
        df['exp'] = np.exp(y)
        df['log'] = np.log(y)
        return train_test_split(df, y, test_size=0.5)

    def test_call(self):
        X_train, X_test, y_train, y_test = self.create_mock_dataset()

        lr = DecisionTreeRegressor(max_depth=2)
        lr.fit(X_train, y_train)

        nof = NumOfFeatures()
        score = nof.call(trial=None,  estimator=lr, X_test=X_test, y_test=y_test)

        assert score < 1  # only 2 features used

        features = nof.get_used_features(trial=None, estimator=lr, X_test=X_test, y_test=y_test)
        assert 'log' in set(features) or 'exp' in set(features)
