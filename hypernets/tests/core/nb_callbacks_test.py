import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# from hypernets.core import nb_callbacks
#
#
# class Test_NB_Callbacks:
#
#     def test_get_importances(self):
#         iris = load_iris()
#         data = iris.data
#         target = iris.target
#         X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
#
#         gbm = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=20)
#         gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='logloss', early_stopping_rounds=2)
#
#         imps = nb_callbacks.extract_importances(gbm)
#         values_type = list(set(map(lambda v: type(v), imps.values())))
#
#         assert len(imps.keys()) == 4
#         assert len(values_type) == 1
#         assert values_type[0] == int or values_type[0] == float  # not numpy type
#         assert json.dumps(imps)
