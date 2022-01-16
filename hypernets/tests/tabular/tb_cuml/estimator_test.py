# -*- coding:utf-8 -*-
"""

"""
import os
import shutil

import pytest

from hypernets.tabular import get_tool_box
from hypernets.tabular.datasets import dsutils
from hypernets.utils import const
from . import if_cuml_ready, is_cuml_installed
from ... import test_output_dir

if is_cuml_installed:
    import cudf
    import cuml


def fit_evaluate(est, data_df, target='y', test_size=0.3, **kwargs):
    tb = get_tool_box(data_df)
    X = data_df
    y = X.pop(target)
    X_train, X_test, y_train, y_test = tb.train_test_split(X, y, test_size=test_size, random_state=335)
    est.fit(X_train, y_train)

    pred = est.predict(X_test)
    if hasattr(est, 'classes_') and len(est.classes_) > 1:
        proba = est.predict_proba(X_test)
        metrics = ['accuracy', 'logloss', ]
        task = const.TASK_BINARY if len(est.classes_) == 2 else const.TASK_MULTICLASS
    else:
        proba = None
        metrics = ['mse', 'mae', 'rmse', 'r2']  # 'msle',
        task = const.TASK_REGRESSION
    scores = tb.metrics.calc_score(y_test, pred, proba, metrics=metrics, task=task)
    print(type(est).__name__, target, scores)


@if_cuml_ready
class TestCumlEstimator:
    work_dir = f'{test_output_dir}/TestCumlEstimator'

    @classmethod
    def setup_class(cls):
        df = dsutils.load_bank()
        df = get_tool_box(df).general_preprocessor(df).fit_transform(df)
        cls.bank_data = df
        cls.bank_data_cudf = cudf.from_pandas(df)
        #
        # cls.boston_data = dsutils.load_blood()
        # cls.boston_data_cudf = cudf.from_pandas(cls.boston_data)
        #
        # cls.movie_lens = dsutils.load_movielens()

        os.makedirs(cls.work_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    @pytest.mark.xfail(reason='TypeError: GPU based predict only accepts np.float32 data. ')
    def test_fail_random_forest_binary_float64(self):
        lr = cuml.ensemble.RandomForestClassifier()
        fit_evaluate(lr, self.bank_data_cudf.copy())

    def test_random_forest_binary(self):
        lr = cuml.ensemble.RandomForestClassifier()
        fit_evaluate(lr, self.bank_data_cudf.astype('float32'))
        # fit_evaluate(lr, self.bank_data_cudf.copy())

    def test_random_forest_multicalss(self):
        lr = cuml.ensemble.RandomForestClassifier()
        fit_evaluate(lr, self.bank_data_cudf.astype('float32'), target='education')

    def test_random_forest_regression(self):
        lr = cuml.ensemble.RandomForestRegressor()
        fit_evaluate(lr, self.bank_data_cudf.astype('float32'), target='duration')

    def test_adapted_random_forest_binary(self):
        from hypernets.tabular.cuml_ex._estimator import AdaptedRandomForestClassifier
        lr = AdaptedRandomForestClassifier()
        # fit_evaluate(lr, self.bank_data_cudf.astype('float32'))
        fit_evaluate(lr, self.bank_data_cudf.copy())

    def test_adapted_random_forest_multicalss(self):
        from hypernets.tabular.cuml_ex._estimator import AdaptedRandomForestClassifier
        lr = AdaptedRandomForestClassifier()
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='education')

    def test_adapted_random_forest_regression(self):
        from hypernets.tabular.cuml_ex._estimator import AdaptedRandomForestRegressor
        lr = AdaptedRandomForestRegressor()
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='duration')

    def test_linear_regression(self):
        lr = cuml.linear_model.LinearRegression()
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='duration')

    def test_ligistic_regression_binary(self):
        lr = cuml.linear_model.LogisticRegression()
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='y')

    def test_ligistic_regression_multiclass(self):
        lr = cuml.linear_model.LogisticRegression()
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='education')

    def test_svc_binary(self):
        lr = cuml.svm.SVC(probability=True)
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='y')

    def test_knn_binary(self):
        lr = cuml.neighbors.KNeighborsClassifier()
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='y')

    def test_knn_multiclass(self):
        lr = cuml.neighbors.KNeighborsClassifier()
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='education')

    def test_knn_regression(self):
        lr = cuml.neighbors.KNeighborsRegressor()
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='duration')

    def test_xgb_binary(self):
        import xgboost
        lr = xgboost.XGBClassifier(use_label_encoder=False)
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='y')

    def test_xgb_multiclass(self):
        import xgboost
        lr = xgboost.XGBClassifier(use_label_encoder=False)
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='education')

    def test_xgb_regression(self):
        import xgboost
        lr = xgboost.XGBRegressor()
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='duration')

    def test_adapted_xgb_regression(self):
        from hypernets.tabular.cuml_ex._estimator import AdaptedXGBRegressor
        lr = AdaptedXGBRegressor()
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='duration')

    def test_adapted_xgb_binary(self):
        from hypernets.tabular.cuml_ex._estimator import AdaptedXGBClassifier
        lr = AdaptedXGBClassifier(use_label_encoder=False)
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='y')

    def test_adapted_xgb_multiclass(self):
        from hypernets.tabular.cuml_ex._estimator import AdaptedXGBClassifier
        lr = AdaptedXGBClassifier(use_label_encoder=False)
        fit_evaluate(lr, self.bank_data_cudf.copy(), target='education')
