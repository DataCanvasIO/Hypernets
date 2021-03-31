# -*- coding:utf-8 -*-
"""

"""
import re
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted

from hypernets.utils import logging, infer_task_type
from .column_selector import column_skewness_kurtosis, column_int, column_object_category_bool

logger = logging.get_logger(__name__)


def root_mean_squared_error(y_true, y_pred,
                            sample_weight=None,
                            multioutput='uniform_average', squared=True):
    return np.sqrt(
        mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput, squared=squared))


def subsample(X, y, max_samples, train_samples, task, random_state=9527):
    stratify = None
    if X.shape[0] > max_samples:
        if task != 'regression':
            stratify = y
        X_train, _, y_train, _ = train_test_split(
            X, y, train_size=max_samples, shuffle=True, stratify=stratify
        )
        if task != 'regression':
            stratify = y_train

        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, train_size=train_samples, shuffle=True, stratify=stratify, random_state=random_state
        )
    else:
        if task != 'regression':
            stratify = y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.5, shuffle=True, stratify=stratify
        )

    return X_train, X_test, y_train, y_test


class SafeLabelEncoder(LabelEncoder):
    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        unseen = len(self.classes_)
        y = np.array([np.searchsorted(self.classes_, x) if x in self.classes_ else unseen for x in y])
        return y


class MultiLabelEncoder:
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        assert len(X.shape) == 2
        n_features = X.shape[1]
        use_iloc = hasattr(X, 'iloc')
        for n in range(n_features):
            le = SafeLabelEncoder()
            data = X.iloc[:, n] if use_iloc else X[:, n]
            le.fit(data)
            self.encoders[n] = le
        return self

    def transform(self, X):
        assert len(X.shape) == 2
        n_features = X.shape[1]
        assert n_features == len(self.encoders.items())
        for n in range(n_features):
            if isinstance(X, np.ndarray):
                X[:, n] = self.encoders[n].transform(X[:, n])
            elif isinstance(X, pd.DataFrame):
                X.iloc[:, n] = self.encoders[n].transform(X.iloc[:, n])
            else:
                raise NotImplementedError('Not supported type')
        return X


class SafeOrdinalEncoder(OrdinalEncoder):
    __doc__ = r'Adapted from sklearn OrdinalEncoder\n' + OrdinalEncoder.__doc__

    # def fit(self, X, y=None):
    #     super().fit(X, y)
    #     #
    #     # def make_encoder(categories):
    #     #     unseen = len(categories)
    #     #     m = dict(zip(categories, range(unseen)))
    #     #     vf = np.vectorize(lambda x: m[x] if x in m.keys() else unseen)
    #     #     return vf
    #     #
    #     # def make_decoder(categories, dtype):
    #     #     if dtype in (np.float32, np.float64, np.float):
    #     #         default_value = np.nan
    #     #     elif dtype in (np.int32, np.int64, np.int, np.uint32, np.uint64, np.uint):
    #     #         default_value = -1
    #     #     else:
    #     #         default_value = None
    #     #         dtype = np.object
    #     #     unseen = len(categories)
    #     #     vf = np.vectorize(lambda x: categories[x] if unseen > x >= 0 else default_value,
    #     #                       otypes=[dtype])
    #     #     return vf
    #     #
    #     # self.encoders_ = [make_encoder(cat) for cat in self.categories_]
    #     # self.decoders_ = [make_decoder(cat, X.dtypes[i]) for i, cat in enumerate(self.categories_)]
    #
    #     return self

    def transform(self, X, y=None):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Unexpected type {}".format(type(X)))

        def make_encoder(categories):
            unseen = len(categories)
            m = dict(zip(categories, range(unseen)))
            vf = np.vectorize(lambda x: m[x] if x in m.keys() else unseen)
            return vf

        values = X if isinstance(X, np.ndarray) else X.values
        encoders_ = [make_encoder(cat) for cat in self.categories_]
        result = [encoders_[i](values[:, i]) for i in range(values.shape[1])]

        if isinstance(X, pd.DataFrame):
            assert len(result) == len(X.columns)
            data = {c: result[i] for i, c in enumerate(X.columns)}
            result = pd.DataFrame(data, dtype=self.dtype)
        else:
            result = np.stack(result, axis=1)
            if self.dtype != result.dtype:
                result = result.astype(self.dtype)

        return result

    def inverse_transform(self, X):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Unexpected type {}".format(type(X)))

        def make_decoder(categories, dtype):
            if dtype in (np.float32, np.float64, np.float):
                default_value = np.nan
            elif dtype in (np.int32, np.int64, np.int, np.uint32, np.uint64, np.uint):
                default_value = -1
            else:
                default_value = None
                dtype = np.object
            unseen = len(categories)
            vf = np.vectorize(lambda x: categories[x] if unseen > x >= 0 else default_value,
                              otypes=[dtype])
            return vf

        values = X if isinstance(X, np.ndarray) else X.values
        decoders_ = [make_decoder(cat, cat.dtype) for i, cat in enumerate(self.categories_)]
        result = [decoders_[i](values[:, i]) for i in range(values.shape[1])]

        if isinstance(X, pd.DataFrame):
            assert len(result) == len(X.columns)
            data = {c: result[i] for i, c in enumerate(X.columns)}
            result = pd.DataFrame(data)
        else:
            result = np.stack(result, axis=1)

        return result


class SafeOneHotEncoder(OneHotEncoder):
    def get_feature_names(self, input_features=None):
        """
        Override this method to remove non-alphanumeric chars from feature names
        """

        check_is_fitted(self)
        cats = self.categories_
        if input_features is None:
            input_features = ['x%d' % i for i in range(len(cats))]
        elif len(input_features) != len(self.categories_):
            raise ValueError(
                "input_features should have length equal to number of "
                "features ({}), got {}".format(len(self.categories_),
                                               len(input_features)))

        feature_names = []
        for i in range(len(cats)):
            names = [input_features[i] + '_' + str(idx) + '_' + re.sub('[^A-Za-z0-9_]+', '_', str(t))
                     for idx, t in enumerate(cats[i])]
            if self.drop_idx_ is not None and self.drop_idx_[i] is not None:
                names.pop(self.drop_idx_[i])
            feature_names.extend(names)

        return np.array(feature_names, dtype=object)


class LogStandardScaler:
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.min_values = None

    def fit(self, X, y=None):
        self.X_min_values = np.min(X)
        self.scaler.fit(np.log(X - self.X_min_values + 1))
        return self

    def transform(self, X):
        X = np.log(np.clip(X - self.X_min_values + 1, a_min=1, a_max=None))
        X = self.scaler.transform(X)
        return X


class SkewnessKurtosisTransformer:
    def __init__(self, transform_fn=None, skew_threshold=0.5, kurtosis_threshold=0.5):
        self.columns_ = []
        self.skewness_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        if transform_fn is None:
            transform_fn = np.log
        self.transform_fn = transform_fn

    def fit(self, X, y=None):
        assert len(X.shape) == 2
        self.columns_ = column_skewness_kurtosis(X, skew_threshold=self.skewness_threshold,
                                                 kurtosis_threshold=self.kurtosis_threshold)
        logger.info(f'SkewnessKurtosisTransformer - selected columns:{self.columns_}')
        return self

    def transform(self, X):
        assert len(X.shape) == 2
        if len(self.columns_) > 0:
            try:
                X[self.columns_] = self.transform_fn(X[self.columns_])
            except Exception as e:
                logger.error(e)
        return X


class FeatureSelectionTransformer():
    def __init__(self, task=None, max_train_samples=10000, max_test_samples=10000, max_cols=10000,
                 ratio_select_cols=0.1,
                 n_max_cols=100, n_min_cols=10, reserved_cols=None):
        self.task = task
        if max_cols <= 0:
            max_cols = 10000
        if max_train_samples <= 0:
            max_train_samples = 10000
        if max_test_samples <= 0:
            max_test_samples = 10000

        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples
        self.max_cols = max_cols
        self.ratio_select_cols = ratio_select_cols
        self.n_max_cols = n_max_cols
        self.n_min_cols = n_min_cols
        self.reserved_cols = reserved_cols
        self.scores_ = {}
        self.columns_ = []

    def get_categorical_features(self, X):
        cat_cols = column_object_category_bool(X)
        int_cols = column_int(X)
        for c in int_cols:
            if X[c].min() >= 0 and X[c].max() < np.iinfo(np.int32).max:
                cat_cols.append(c)
        return cat_cols

    def feature_score(self, F_train, y_train, F_test, y_test):
        if self.task is None:
            self.task, _ = infer_task_type(y_train)

        if self.task == 'regression':
            model = LGBMRegressor()
            eval_metric = root_mean_squared_error
        else:
            model = LGBMClassifier()
            eval_metric = log_loss

        cat_cols = self.get_categorical_features(F_train)

        model.fit(F_train, y_train,
                  # eval_set=(F_test, y_test),
                  # early_stopping_rounds=20,
                  # verbose=0,
                  # categorical_feature=cat_cols,
                  # eval_metric=eval_metric,
                  )
        if self.task == 'regression':
            y_pred = model.predict(F_test)
        else:
            y_pred = model.predict_proba(F_test)[:, 1]

        score = eval_metric(y_test, y_pred)
        return score

    def fit(self, X, y):
        start_time = time.time()
        if self.task is None:
            self.task, _ = infer_task_type(y)
        columns = X.columns.to_list()
        logger.info(f'all columns: {columns}')
        if self.reserved_cols is not None:
            self.reserved_cols = list(set(self.reserved_cols).intersection(columns))
            logger.info(f'exclude reserved columns: {self.reserved_cols}')
            columns = list(set(columns) - set(self.reserved_cols))

        if len(columns) > self.max_cols:
            columns = np.random.choice(columns, self.max_cols, replace=False)

        if len(columns) <= 0:
            logger.warn('no columns to score')
            self.columns_ = self.reserved_cols
            self.scores_ = {}
            return self

        X_score = X[columns]
        X_train, X_test, y_train, y_test = subsample(X_score, y,
                                                     max_samples=self.max_test_samples + self.max_train_samples,
                                                     train_samples=self.max_train_samples,
                                                     task=self.task)
        if self.task != 'regression' and y_train.dtype != 'int':
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        cat_cols = column_object_category_bool(X_train)

        if len(cat_cols) > 0:
            logger.info('ordinal encoding...')
            X_train['__datacanvas__source__'] = 'train'
            X_test['__datacanvas__source__'] = 'test'
            X_all = pd.concat([X_train, X_test], axis=0)
            oe = OrdinalEncoder()
            X_all[cat_cols] = oe.fit_transform(X_all[cat_cols]).astype('int')

            X_train = X_all[X_all['__datacanvas__source__'] == 'train']
            X_test = X_all[X_all['__datacanvas__source__'] == 'test']
            X_train.pop('__datacanvas__source__')
            X_test.pop('__datacanvas__source__')

        self.scores_ = {}

        for c in columns:
            F_train = X_train[[c]]
            F_test = X_test[[c]]
            self.scores_[c] = self.feature_score(F_train, y_train, F_test, y_test)
            logger.info(f'Feature score: {c}={self.scores_[c]}')

        sorted_scores = sorted([[col, score] for col, score in self.scores_.items()], key=lambda x: x[1])
        logger.info(f'feature scores:{sorted_scores}')
        topn = np.min([np.max([int(len(columns) * self.ratio_select_cols), np.min([len(columns), self.n_min_cols])]),
                       self.n_max_cols])
        if self.reserved_cols is not None:
            self.columns_ = self.reserved_cols
        else:
            self.columns_ = []
        self.columns_ += [s[0] for s in sorted_scores[:topn]]

        logger.info(f'selected columns:{self.columns_}')
        logger.info(f'taken {time.time() - start_time}s')

        del X_score, X_train, X_test, y_train, y_test

        return self

    def transform(self, X):
        return X[self.columns_]


class FloatOutputImputer(SimpleImputer):

    def transform(self, X):
        return super().transform(X).astype(np.float64)
