import time

import pandas as pd
from sklearn.compose import make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
import sklearn
import xgboost as xgb

X_train = pd.read_csv("/Users/wuhf/PycharmProjects/cooka.benchmark/dataset/binary/Santander_Customer/train.csv").sample(2000)
y_train = X_train.pop("TARGET")
print(f"train data shape: {X_train.shape}")

X_test = pd.read_csv("/Users/wuhf/PycharmProjects/cooka.benchmark/dataset/binary/Santander_Customer/test.csv")
y_test = X_test.pop("TARGET")
print(f"test data shape: {X_test.shape}")

category_cols = make_column_selector(dtype_include=object)(X_train)
dfm = DataFrameMapper([(c, LabelEncoder()) for c in category_cols], input_df=True, df_out=True, default=None)
X_train = dfm.fit_transform(X_train)
X_test = dfm.transform(X_test)


clf = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=2019
)

t1 = time.time()
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=['error', 'auc'])
print(f"escaped: {time.time()-t1}")
y_score = clf.predict_proba(X_test)  # clf._Booster.feature_names
y_pred = clf.predict(X_test)

roc_auc_score = sklearn.metrics.roc_auc_score(y_test.values, y_score[:, 1], labels=clf.classes_)
print(f"roc_auc_score: {roc_auc_score}")
