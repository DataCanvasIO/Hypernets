import time
import pandas as pd
from sklearn.compose import make_column_selector
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
import sklearn
import lightgbm as lgb


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

clf = lgb.LGBMClassifier(learning_rate=0.1,
                         n_estimators=300,
                         gpu_platform_id=0,
                         gpu_device_id=0)
t1 = time.time()
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# feature_name_

print(f"escaped: {time.time()-t1}")
y_score = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

roc_auc_score = sklearn.metrics.roc_auc_score(y_test.values, y_score[:, 1], labels=clf.classes_)
print(f"roc_auc_score: {roc_auc_score}")
