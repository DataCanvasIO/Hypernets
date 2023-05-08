# -*- coding:utf-8 -*-
"""

"""
import inspect
import re
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted

from hypernets.tabular import column_selector
from hypernets.utils import logging, const
from . import tb_transformer, get_tool_box

try:
    import jieba

    _jieba_installed = True
except ImportError:
    _jieba_installed = False

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


@tb_transformer(pd.DataFrame)
class PassThroughEstimator(BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


@tb_transformer(pd.DataFrame)
class AsTypeTransformer(BaseEstimator):
    def __init__(self, *, dtype):
        assert dtype is not None
        self.dtype = dtype

        super(AsTypeTransformer, self).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(self.dtype)

    def fit_transform(self, X, y=None):
        return self.transform(X)


# class SafeLabelEncoder(LabelEncoder):
#     def transform(self, y):
#         check_is_fitted(self, 'classes_')
#         y = column_or_1d(y, warn=True)
#
#         unseen = len(self.classes_)
#         y = np.array([np.searchsorted(self.classes_, x) if x in self.classes_ else unseen for x in y])
#         return y

@tb_transformer(pd.DataFrame)
class ConstantImputer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_values=np.nan, fill_value=None, copy=True) -> None:
        super().__init__()

        self.missing_values = missing_values
        self.fill_value = fill_value
        self.copy = copy

    def fit(self, X, y=None, ):
        return self

    def transform(self, X, y=None):
        if self.copy:
            X = X.copy()

        X.replace(self.missing_values, self.fill_value, inplace=True)
        return X


@tb_transformer(pd.DataFrame, name='SimpleImputer')
class SafeSimpleImputer(SimpleImputer):
    """
    passthrough bool columns
    """

    def fit(self, X, y=None, ):
        if isinstance(X, pd.DataFrame):
            bool_cols = X.select_dtypes(include='bool').columns.tolist()
            if bool_cols:
                df_notbool = X.select_dtypes(exclude='bool')
                if df_notbool.shape[1] > 0:
                    super().fit(df_notbool, y=y)
                self.bool_cols_ = bool_cols
            else:
                super().fit(X, y=y)
        else:
            super().fit(X, y=y)

        return self

    def transform(self, X):
        bool_cols = getattr(self, 'bool_cols_', None)
        if bool_cols is not None:
            assert isinstance(X, pd.DataFrame)
            not_bools = [c for c in X.columns.tolist() if c not in bool_cols]
            Xt = super().transform(X[not_bools])
            X = X.copy()
            X[not_bools] = Xt
            return X if isinstance(Xt, pd.DataFrame) else X.values
        else:
            return super().transform(X)


@tb_transformer(pd.DataFrame)
class SafeLabelEncoder(LabelEncoder):
    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        unseen = len(self.classes_)
        lookup_table = dict(zip(self.classes_, list(range(0, unseen))))
        out = np.full(len(y), unseen)
        ind_id = 0
        for cell_value in y:
            if cell_value in lookup_table:
                out[ind_id] = lookup_table[cell_value]
            ind_id += 1
        return out


@tb_transformer(pd.DataFrame)
class MultiLabelEncoder(BaseEstimator):
    def __init__(self, columns=None, dtype=None):
        super(MultiLabelEncoder, self).__init__()

        self.columns = columns
        self.dtype = dtype

        # fitted
        self.encoders = {}

    def fit(self, X, y=None):
        assert len(X.shape) == 2
        assert isinstance(X, pd.DataFrame) or self.columns is None

        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                self.columns = X.columns.tolist()
            for col in self.columns:
                data = X[col]
                if data.dtype == 'object':
                    data = data.astype('str')
                    # print(f'Column "{col}" has been convert to "str" type.')
                le = SafeLabelEncoder()
                le.fit(data)
                self.encoders[col] = le
        else:
            n_features = X.shape[1]
            for n in range(n_features):
                data = X[:, n]
                le = SafeLabelEncoder()
                le.fit(data)
                self.encoders[n] = le

        return self

    def transform(self, X):
        assert len(X.shape) == 2
        assert isinstance(X, pd.DataFrame) or self.columns is None

        if self.columns is not None:  # dataframe
            for col in self.columns:
                data = X[col]
                if data.dtype == 'object':
                    data = data.astype('str')
                data_t = self.encoders[col].transform(data)
                if self.dtype:
                    data_t = data_t.astype(self.dtype)
                X[col] = data_t
        else:
            n_features = X.shape[1]
            assert n_features == len(self.encoders.items())
            for n in range(n_features):
                X[:, n] = self.encoders[n].transform(X[:, n])
            if self.dtype:
                X = X.astype(self.dtype)

        return X

    def fit_transform(self, X, *args):
        assert len(X.shape) == 2
        assert isinstance(X, pd.DataFrame) or self.columns is None

        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                self.columns = X.columns.tolist()
            for col in self.columns:
                data = X[col]
                if data.dtype == 'object':
                    data = data.astype('str')
                    # print(f'Column "{col}" has been convert to "str" type.')
                le = SafeLabelEncoder()
                data_t = le.fit_transform(data)
                if self.dtype:
                    data_t = data_t.astype(self.dtype)
                X[col] = data_t
                self.encoders[col] = le
        else:
            n_features = X.shape[1]
            for n in range(n_features):
                data = X[:, n]
                le = SafeLabelEncoder()
                X[:, n] = le.fit_transform(data)
                self.encoders[n] = le
            if self.dtype:
                X = X.astype(self.dtype)

        return X


@tb_transformer(pd.DataFrame)
class SafeOrdinalEncoder(OrdinalEncoder):
    __doc__ = r'Adapted from sklearn OrdinalEncoder\n' + OrdinalEncoder.__doc__

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
            if dtype in (np.float32, np.float64, float):
                default_value = np.nan
            elif dtype in (np.int32, np.int64, np.uint32, np.uint64, np.uint, int):
                default_value = -1
            else:
                default_value = None
                dtype = object
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


@tb_transformer(pd.DataFrame)
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


@tb_transformer(pd.DataFrame)
class LogStandardScaler(BaseEstimator):
    def __init__(self, copy=True, with_mean=True, with_std=True):
        super(LogStandardScaler, self).__init__()

        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.min_values = None
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, X, y=None):
        self.X_min_values = np.min(X)
        self.scaler.fit(np.log(X - self.X_min_values + 1))
        return self

    def transform(self, X):
        X = np.log(np.clip(X - self.X_min_values + 1, a_min=1, a_max=None))
        X = self.scaler.transform(X)
        return X


@tb_transformer(pd.DataFrame)
class SkewnessKurtosisTransformer(BaseEstimator):
    def __init__(self, transform_fn=None, skew_threshold=0.5, kurtosis_threshold=0.5):
        self.columns_ = []
        self.skewness_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        if transform_fn is None:
            transform_fn = np.log
        self.transform_fn = transform_fn

    def fit(self, X, y=None):
        assert len(X.shape) == 2
        self.columns_ = column_selector.column_skewness_kurtosis(X, skew_threshold=self.skewness_threshold,
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


@tb_transformer(pd.DataFrame)
class FeatureSelectionTransformer(BaseEstimator):
    def __init__(self, task=None, max_train_samples=10000, max_test_samples=10000, max_cols=10000,
                 ratio_select_cols=0.1,
                 n_max_cols=100, n_min_cols=10, reserved_cols=None):
        super(FeatureSelectionTransformer, self).__init__()

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
        cat_cols = column_selector.column_object_category_bool(X)
        int_cols = column_selector.column_int(X)
        for c in int_cols:
            if X[c].min() >= 0 and X[c].max() < np.iinfo(np.int32).max:
                cat_cols.append(c)
        return cat_cols

    def feature_score(self, F_train, y_train, F_test, y_test):
        if self.task is None:
            self.task, _ = get_tool_box(y_train).infer_task_type(y_train)

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
            self.task, _ = get_tool_box(y).infer_task_type(y)
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

        cat_cols = column_selector.column_object_category_bool(X_train)

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


@tb_transformer(pd.DataFrame)
class FeatureImportancesSelectionTransformer(BaseEstimator):
    def __init__(self, task=None, strategy=None, threshold=None, quantile=None, number=None, data_clean=True):
        super().__init__()

        self.task = task
        self.strategy = strategy
        self.threshold = threshold
        self.quantile = quantile
        self.number = number
        self.data_clean = data_clean

        # fitted
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self.selected_features_ = None

    def fit(self, X, y):
        tb = get_tool_box(X, y)
        if self.task is None:
            self.task, _ = tb.infer_task_type(y)

        columns_in = X.columns.to_list()
        # logger.info(f'all columns: {columns}')

        if self.data_clean:
            logger.info('data cleaning')
            kwargs = dict(replace_inf_values=np.nan, drop_label_nan_rows=True,
                          drop_constant_columns=True, drop_duplicated_columns=False,
                          drop_idness_columns=True, reduce_mem_usage=False,
                          correct_object_dtype=False, int_convert_to=None,
                          )
            dc = tb.data_cleaner(**kwargs)
            X, y = dc.fit_transform(X, y)
            assert set(X.columns.tolist()).issubset(set(columns_in))

        preprocessor = tb.general_preprocessor(X)
        estimator = tb.general_estimator(X, y, task=self.task)

        if self.task != 'regression' and y.dtype != 'int':
            logger.info('label encoding')
            le = tb.transformers['LabelEncoder']()
            y = le.fit_transform(y)

        logger.info('preprocessing')
        X = preprocessor.fit_transform(X, y)
        logger.info('scoring')
        estimator.fit(X, y)
        importances = estimator.feature_importances_

        selected, unselected = \
            tb.select_feature_by_importance(importances, strategy=self.strategy,
                                            threshold=self.threshold,
                                            quantile=self.quantile,
                                            number=self.number)
        columns = X.columns.to_list()
        selected = [columns[i] for i in selected]

        if len(columns) != len(columns_in):
            importances = [0.0 if c not in columns else importances[columns.index(c)] for c in columns_in]
            importances = np.array(importances)

        self.n_features_in_ = len(columns_in)
        self.feature_names_in_ = columns_in
        self.feature_importances_ = importances
        self.selected_features_ = selected

        # logger.info(f'selected columns:{self.selected_features_}')

        return self

    def transform(self, X):
        return X[self.selected_features_]


@tb_transformer(pd.DataFrame)
class FloatOutputImputer(SimpleImputer):

    def transform(self, X):
        return super().transform(X).astype(np.float64)


@tb_transformer(pd.DataFrame)
class LgbmLeavesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_vars, cont_vars, task, **params):
        super(LgbmLeavesEncoder, self).__init__()

        self.lgbm = None
        self.cat_vars = cat_vars
        self.cont_vars = cont_vars
        self.new_columns = []
        self.task = task
        self.lgbm_params = params

    def fit(self, X, y):
        from lightgbm import LGBMClassifier, LGBMRegressor

        X[self.cont_vars] = X[self.cont_vars].astype('float')
        X[self.cat_vars] = X[self.cat_vars].astype('int')

        logger.info(f'LightGBM task:{self.task}')
        if self.task == const.TASK_MULTICLASS:  # multiclass label
            if len(y.shape) > 1 and y.shape[1] > 1:
                num_class = y.shape[-1]
                if self.lgbm_params is None:
                    self.lgbm_params = {}
                y = y.argmax(axis=-1)
            else:
                if hasattr(y, 'unique'):
                    num_class = len(set(y.unique()))
                else:
                    num_class = len(set(y))
            self.lgbm_params['num_class'] = num_class + 1
            self.lgbm_params['n_estimators'] = int(100 / num_class) + 1

        if self.task == const.TASK_REGRESSION:
            self.lgbm = LGBMRegressor(**self.lgbm_params)
        else:
            self.lgbm = LGBMClassifier(**self.lgbm_params)
        self.lgbm.fit(X, y)
        return self

    def transform(self, X):
        X[self.cont_vars] = X[self.cont_vars].astype('float')
        X[self.cat_vars] = X[self.cat_vars].astype('int')

        leaves = self.lgbm.predict(X, pred_leaf=True, num_iteration=self.lgbm.best_iteration_)
        new_columns = [f'lgbm_leaf_{i}' for i in range(leaves.shape[1])]
        df_leaves = pd.DataFrame(leaves, columns=new_columns, index=X.index)
        result = pd.concat([X, df_leaves], axis=1)

        self.new_columns = new_columns

        return result


@tb_transformer(pd.DataFrame)
class CategorizeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, remain_numeric=True):
        super(CategorizeEncoder, self).__init__()

        self.columns = columns
        self.remain_numeric = remain_numeric

        # fitted
        self.new_columns = []

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns.tolist()

        new_columns = []
        if self.remain_numeric:
            for col in self.columns:
                target_col = col + const.COLUMNNAME_POSTFIX_CATEGORIZE
                new_columns.append((target_col, 'str', X[col].nunique()))

        self.new_columns = new_columns

        return self

    def transform(self, X):
        for col in self.columns:
            if self.remain_numeric:
                target_col = col + const.COLUMNNAME_POSTFIX_CATEGORIZE
            else:
                target_col = col
            X[target_col] = X[col].astype('str')
        return X


@tb_transformer(pd.DataFrame)
class MultiKBinsDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, bins=None, strategy='quantile'):
        super(MultiKBinsDiscretizer, self).__init__()

        logger.info(f'{len(columns)} variables to discrete.')
        self.columns = columns
        self.bins = bins
        self.strategy = strategy
        self.new_columns = []
        self.encoders = {}

    def fit(self, X, y=None):
        self.new_columns = []
        if self.columns is None:
            self.columns = X.columns.tolist()
        for col in self.columns:
            new_name = col + const.COLUMNNAME_POSTFIX_DISCRETE
            n_unique = X[col].nunique()
            # n_null = X[col].isnull().sum()
            c_bins = self.bins
            if c_bins is None or c_bins <= 0:
                c_bins = round(n_unique ** 0.25) + 1
            encoder = KBinsDiscretizer(n_bins=c_bins, encode='ordinal', strategy=self.strategy)
            self.new_columns.append((col, new_name, encoder.n_bins))
            encoder.fit(X[[col]])
            self.encoders[col] = encoder
        return self

    def transform(self, X):
        for col in self.columns:
            new_name = col + const.COLUMNNAME_POSTFIX_DISCRETE
            encoder = self.encoders[col]
            nc = encoder.transform(X[[col]]).astype(const.DATATYPE_LABEL).reshape(-1)
            X[new_name] = nc
        return X


@tb_transformer(pd.DataFrame)
class DataFrameWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transform, columns=None):
        super(DataFrameWrapper, self).__init__()

        self.transformer = transform
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns.tolist()
        self.transformer.fit(X)
        return self

    def transform(self, X):
        df = pd.DataFrame(self.transformer.transform(X))
        df.columns = self.columns
        return df


@tb_transformer(pd.DataFrame)
class GaussRankScaler(BaseEstimator):
    def __init__(self):
        super(GaussRankScaler, self).__init__()

        self.epsilon = 0.001
        self.lower = -1 + self.epsilon
        self.upper = 1 - self.epsilon
        self.range = self.upper - self.lower
        self.divider = None

    def fit_transform(self, X, y=None):
        from scipy.special import erfinv
        i = np.argsort(X, axis=0)
        j = np.argsort(i, axis=0)

        assert (j.min() == 0).all()
        assert (j.max() == len(j) - 1).all()

        j_range = len(j) - 1
        self.divider = j_range / self.range

        transformed = j / self.divider
        transformed = transformed - self.upper
        transformed = erfinv(transformed)

        return transformed


@tb_transformer(pd.DataFrame)
class VarLenFeatureEncoder:
    def __init__(self, sep='|'):
        super(VarLenFeatureEncoder, self).__init__()

        self.sep = sep
        self.encoder: SafeLabelEncoder = None
        self._max_element_length = 0

    def fit(self, X: pd.Series):
        self._max_element_length = 0  # reset
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        key_set = set()
        # flat map
        for keys in X.map(lambda _: _.split(self.sep)):
            if len(keys) > self._max_element_length:
                self._max_element_length = len(keys)
            key_set.update(keys)
        key_set = list(key_set)
        key_set.sort()

        lb = SafeLabelEncoder()  # fix unseen values
        lb.fit(np.array(key_set))
        self.encoder = lb
        return self

    def transform(self, X: pd.Series):
        if self.encoder is None:
            raise RuntimeError("Not fit yet .")

        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
        data = X.map(lambda _: (self.encoder.transform(_.split(self.sep)) + 1).tolist())

        transformed = self.pad_sequences(data, maxlen=self._max_element_length, padding='post',
                                         truncating='post').tolist()  # cut last elements
        return transformed

    @property
    def n_classes(self):
        return len(self.encoder.classes_)

    @property
    def max_element_length(self):
        return self._max_element_length

    @staticmethod
    def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
        """Adapted from tensorflow.python.keras.preprocessing.sequence.pad_sequences
        """
        if not hasattr(sequences, '__len__'):
            raise ValueError('`sequences` must be iterable.')
        num_samples = len(sequences)

        lengths = []
        sample_shape = ()
        flag = True

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.

        for x in sequences:
            try:
                lengths.append(len(x))
                if flag and len(x):
                    sample_shape = np.asarray(x).shape[1:]
                    flag = False
            except TypeError:
                raise ValueError('`sequences` must be a list of iterables. '
                                 'Found non-iterable: ' + str(x))

        if maxlen is None:
            maxlen = np.max(lengths)

        is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
        if isinstance(value, (str, bytes)) and dtype != object and not is_dtype_str:
            raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                             "You should set `dtype=object` for variable length strings."
                             .format(dtype, type(value)))

        x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
        for idx, s in enumerate(sequences):
            if not len(s):
                continue  # empty list/array was found
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" '
                                 'not understood' % truncating)

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence at position %s '
                                 'is different from expected shape %s' %
                                 (trunc.shape[1:], idx, sample_shape))

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        return x


@tb_transformer(pd.DataFrame)
class MultiVarLenFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        super(MultiVarLenFeatureEncoder, self).__init__()

        self.features = features

        # fitted
        self.encoders_ = {}  # feature name -> VarLenFeatureEncoder
        self.max_length_ = {}  # feature name -> max length

    def fit(self, X, y=None):
        encoders = {feature[0]: VarLenFeatureEncoder(feature[1]) for feature in self.features}
        max_length = {}

        for k, v in encoders.items():
            v.fit(X[k])
            max_length[k] = v.max_element_length

        self.encoders_ = encoders
        self.max_length_ = max_length

        return self

    def transform(self, X):
        for k, v in self.encoders_.items():
            X[k] = v.transform(X[k])
        return X


@tb_transformer(pd.DataFrame)
class LocalizedTfidfVectorizer(TfidfVectorizer):
    def decode(self, doc):
        doc = super().decode(doc)

        if _jieba_installed and self._exist_chinese(doc):
            doc = ' '.join(jieba.cut(doc))

        return doc

    @staticmethod
    def _exist_chinese(s):
        if isinstance(s, str):
            for ch in s:
                if u'\u4e00' <= ch <= u'\u9fff':
                    return True

        return False


@tb_transformer(pd.DataFrame)
class TfidfEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, flatten=False, **kwargs):
        assert columns is None or isinstance(columns, (str, list, tuple))
        if isinstance(columns, str):
            columns = [columns]

        super(TfidfEncoder, self).__init__()

        self.columns = columns
        self.flatten = flatten
        self.encoder_kwargs = kwargs.copy()

        # fitted
        self.encoders_ = None

    def create_encoder(self):
        return LocalizedTfidfVectorizer(**self.encoder_kwargs)

    def fit(self, X, y=None):
        assert len(X.shape) == 2

        if self.columns is None:
            if hasattr(X, 'columns'):
                columns = column_selector.column_object(X)
            else:
                columns = range(X.shape[1])
        else:
            columns = self.columns

        encoders = {}
        for c in columns:
            encoder = self.create_encoder()
            Xc = X[c] if hasattr(X, 'columns') else X[:, c]
            encoders[c] = encoder.fit(Xc)

        self.encoders_ = encoders

        return self

    def transform(self, X, y=None):
        assert self.encoders_ is not None
        assert len(X.shape) == 2

        from . import get_tool_box
        tb = get_tool_box(X)

        if hasattr(X, 'columns'):
            X = X.copy()
            if self.flatten:
                dfs = []
                encoded = []
                for c, encoder in self.encoders_.items():
                    t = encoder.transform(X[c]).toarray()
                    dfs.append(tb.array_to_df(t, index=X.index, columns=[f'{c}_tfidf_{i}' for i in range(t.shape[1])]))
                    encoded.append(c)
                unencoded = set(X.columns.tolist()) - set(encoded)
                if len(unencoded) > 0:
                    dfs.insert(0, X[unencoded])
                X = tb.concat_df(dfs, axis=1)
            else:
                for c, encoder in self.encoders_.items():
                    t = encoder.transform(X[c]).toarray()
                    X[c] = t.tolist()
        else:
            r = []
            for i in range(X.shape[1]):
                Xi = X[:, i]
                if i in self.encoders_.keys():
                    encoder = self.encoders_[i]
                    t = encoder.transform(Xi).toarray()
                    if not self.flatten:
                        t = tb.collapse_last_dim(t, keep_dim=True)
                    r.append(t)
                else:
                    r.append(Xi)
            X = tb.hstack_array(r)

        return X


@tb_transformer(pd.DataFrame)
class DatetimeEncoder(BaseEstimator, TransformerMixin):
    all_items = ['year', 'month', 'day', 'hour', 'minute', 'second',
                 'week', 'weekday', 'dayofyear',
                 'timestamp']
    all_items = {k: k for k in all_items}
    all_items['timestamp'] = lambda x: (x.astype('int64') * 1e-9)

    default_include = ['month', 'day', 'hour', 'minute',
                       'week', 'weekday', 'dayofyear']

    def __init__(self, columns=None, include=None, exclude=None, extra=None, drop_constants=True):
        assert columns is None or isinstance(columns, (str, list, tuple))
        assert include is None or isinstance(include, (str, list, tuple))
        assert exclude is None or isinstance(exclude, (str, list, tuple))
        assert extra is None or isinstance(extra, (tuple, list))
        if extra is not None:
            assert all(len(x) == 2 and isinstance(x[0], str)
                       and (x[1] is None or isinstance(x[1], str) or callable(x[1]))
                       for x in extra)

        if isinstance(columns, str):
            columns = [columns]

        if include is None:
            to_extract = self.default_include
        elif isinstance(include, str):
            to_extract = [include]
        else:
            to_extract = include

        if isinstance(exclude, str):
            exclude = [exclude]
        if exclude is not None:
            to_extract = [i for i in to_extract if i not in exclude]

        assert all(i in self.all_items for i in to_extract)

        to_extract = {k: self.all_items[k] for k in to_extract}
        if isinstance(extra, (tuple, list)):
            for k, c in extra:
                to_extract[k] = c

        super(DatetimeEncoder, self).__init__()

        self.columns = columns
        self.include = include
        self.exclude = exclude
        self.extra = extra
        self.drop_constants = drop_constants

        self.extract_ = to_extract

    def fit(self, X, y=None):
        if self.columns is None:
            X = self.to_dataframe(X)
            self.columns = column_selector.column_all_datetime(X)

        return self

    def transform(self, X, y=None):
        if len(self.columns) > 0:
            X_orig = X
            X = self.to_dataframe(X)
            input_df = X_orig is X

            dfs = [df for c in self.columns for df in self.transform_column(X[c])]
            unencoded = set(X.columns.tolist()) - set(self.columns)
            X = X[list(unencoded)]
            if len(dfs) > 0:
                if len(unencoded) > 0:
                    dfs.insert(0, X)
                tb = get_tool_box(*dfs)
                X = tb.concat_df(dfs, axis=1)

            if not input_df:
                X = X.values

        return X

    @staticmethod
    def to_dataframe(X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X

    def transform_column(self, Xc):
        assert getattr(Xc, 'dt', None) is not None
        dfs = []

        for k, c in self.extract_.items():
            if c is None:
                c = k
            if isinstance(c, str):
                if hasattr(Xc.dt, c):
                    t = getattr(Xc.dt, c)
                else:
                    continue
            else:
                t = c(Xc)
            t.name = f'{Xc.name}_{k}'
            dfs.append(t)

        if self.drop_constants and len(Xc) > 1:
            dfs = [t for t in dfs if t.nunique() > 1]

        return dfs


class TargetEncoder(BaseEstimator):
    """
    Adapted from cuml.preprocessing.TargetEncoder
    """

    def __init__(self, n_folds=4, smooth=0, seed=42, split_method='interleaved'):
        if smooth < 0:
            raise ValueError(f'smooth {smooth} is not zero or positive')
        if n_folds < 0 or not isinstance(n_folds, int):
            raise ValueError(
                'n_folds {} is not a postive integer'.format(n_folds))

        if not isinstance(seed, int):
            raise ValueError('seed {} is not an integer'.format(seed))

        if split_method not in {'random', 'continuous', 'interleaved'}:
            msg = ("split_method should be either 'random'"
                   " or 'continuous' or 'interleaved', "
                   "got {0}.".format(self.split))
            raise ValueError(msg)

        self.n_folds = n_folds
        self.seed = seed
        self.smooth = smooth
        self.split = split_method
        self.y_col = '__TARGET__'
        self.x_col = '__FEA__'
        self.out_col = '__TARGET_ENCODE__'
        self.fold_col = '__FOLD__'
        self.id_col = '__INDEX__'

        # fitted
        self._fitted = False
        self.train = None
        self.train_encode = None
        self.mean = None
        self.encode_all = None

    def fit(self, x, y):
        """
        Fit a TargetEncoder instance to a set of categories

        Parameters
        ----------
        x: cudf.Series or cudf.DataFrame or cupy.ndarray
           categories to be encoded. It's elements may or may
           not be unique
        y : cudf.Series or cupy.ndarray
            Series containing the target variable.

        Returns
        -------
        self : TargetEncoder
            A fitted instance of itself to allow method chaining
        """
        res, train = self._fit_transform(x, y)
        self.train_encode = res
        self.train = train
        self._fitted = True
        return self

    def fit_transform(self, x, y):
        """
        Simultaneously fit and transform an input

        This is functionally equivalent to (but faster than)
        `TargetEncoder().fit(y).transform(y)`
        """
        self.fit(x, y)
        return self.train_encode

    def transform(self, x):
        """
        Transform an input into its categorical keys.

        This is intended for test data. For fitting and transforming
        the training data, prefer `fit_transform`.

        Parameters
        ----------
        x : cudf.Series
            Input keys to be transformed. Its values doesn't have to
            match the categories given to `fit`

        Returns
        -------
        encoded : cupy.ndarray
            The ordinally encoded input series

        """
        self._check_is_fitted()
        test = self._to_dataframe(x)
        if self._is_train_df(test):
            return self.train_encode
        x_cols = [i for i in test.columns.tolist() if i != self.id_col]
        test = test.merge(self.encode_all, on=x_cols, how='left')
        return self._impute_and_sort(test)

    def _fit_transform(self, x, y):
        """
        Core function of target encoding
        """
        np.random.seed(self.seed)
        train = self._to_dataframe(x)
        x_cols = [i for i in train.columns.tolist() if i != self.id_col]
        train[self.y_col] = self._make_y_column(y)

        self.n_folds = min(self.n_folds, len(train))
        train[self.fold_col] = self._make_fold_column(len(train))

        self.mean = train[self.y_col].mean()

        y_count_each_fold, y_count_all = self._groupby_agg(train,
                                                           x_cols,
                                                           op='count')

        y_sum_each_fold, y_sum_all = self._groupby_agg(train,
                                                       x_cols,
                                                       op='sum')
        """
        Note:
            encode_each_fold is used to encode train data.
            encode_all is used to encode test data.
        """
        cols = [self.fold_col] + x_cols
        encode_each_fold = self._compute_output(y_sum_each_fold,
                                                y_count_each_fold,
                                                cols,
                                                f'{self.y_col}_x')
        encode_all = self._compute_output(y_sum_all,
                                          y_count_all,
                                          x_cols,
                                          self.y_col)
        self.encode_all = encode_all

        train = train.merge(encode_each_fold, on=cols, how='left')
        del encode_each_fold
        return self._impute_and_sort(train), train

    def _make_y_column(self, y):
        """
        Create a target column given y
        """
        if isinstance(y, pd.Series):
            return y.values
        elif isinstance(y, np.ndarray):
            if len(y.shape) == 1:
                return y
            elif y.shape[1] == 1:
                return y[:, 0]
            else:
                raise ValueError(f"Input of shape {y.shape} "
                                 "is not a 1-D array.")
        else:
            raise TypeError(f"Input of type {type(y)} is not pandas.Series or numpy.ndarray")

    def _make_fold_column(self, len_train):
        """
        Create a fold id column for each split_method
        """
        if self.split == 'random':
            return np.random.randint(0, self.n_folds, len_train)
        elif self.split == 'continuous':
            return (np.arange(len_train) /
                    (len_train / self.n_folds)) % self.n_folds
        elif self.split == 'interleaved':
            return np.arange(len_train) % self.n_folds
        else:
            msg = ("split should be either 'random'"
                   " or 'continuous' or 'interleaved', "
                   "got {0}.".format(self.split))
            raise ValueError(msg)

    def _compute_output(self, df_sum, df_count, cols, y_col):
        """
        Compute the output encoding based on aggregated sum and count
        """
        df_sum = df_sum.merge(df_count, on=cols, how='left')
        smooth = self.smooth
        df_sum[self.out_col] = (df_sum[f'{y_col}_x'] +
                                smooth * self.mean) / \
                               (df_sum[f'{y_col}_y'] +
                                smooth)
        return df_sum

    def _groupby_agg(self, train, x_cols, op):
        """
        Compute aggregated value of each fold and overall dataframe
        grouped by `x_cols` and agg by `op`
        """
        cols = [self.fold_col] + x_cols
        df_each_fold = train.groupby(cols, as_index=False) \
            .agg({self.y_col: op})
        df_all = df_each_fold.groupby(x_cols, as_index=False) \
            .agg({self.y_col: 'sum'})

        df_each_fold = df_each_fold.merge(df_all, on=x_cols, how='left')
        df_each_fold[f'{self.y_col}_x'] = df_each_fold[f'{self.y_col}_y'] - \
                                          df_each_fold[f'{self.y_col}_x']
        return df_each_fold, df_all

    def _check_is_fitted(self):
        if not self._fitted or self.train is None:
            msg = ("This LabelEncoder instance is not fitted yet. Call 'fit' "
                   "with appropriate arguments before using this estimator.")
            raise ValueError(msg)

    def _is_train_df(self, df):
        """
        Return True if the dataframe `df` is the training dataframe, which
        is used in `fit_transform`
        """
        if len(df) != len(self.train):
            return False
        self.train = self.train.sort_values(self.id_col).reset_index(drop=True)
        for col in df.columns:
            if col not in self.train.columns:
                raise ValueError(f"Input column {col} "
                                 "is not in train data.")
            if not (df[col] == self.train[col]).all():
                return False
        return True

    def _impute_and_sort(self, df):
        """
        Impute and sort the result encoding in the same row order as input
        """
        # df[self.out_col] = df[self.out_col].nans_to_nulls()
        df[self.out_col] = df[self.out_col].fillna(self.mean)
        df = df.sort_values(self.id_col)
        res = df[self.out_col].values.copy()

        return res

    def _to_dataframe(self, x):
        if isinstance(x, pd.DataFrame):
            df = x.copy()
        elif isinstance(x, pd.Series):
            df = x.to_frame().copy()
        elif isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                df = pd.DataFrame({self.x_col: x})
            else:
                df = pd.DataFrame(x, columns=[f'{self.x_col}_{i}' for i in range(x.shape[1])])
        else:
            raise TypeError(f"Input of type {type(x)} is pandas.Series or pandas.DataFrame or numpy.ndarray")

        df[self.id_col] = np.arange(len(x))
        return df


@tb_transformer(pd.DataFrame)
class SlimTargetEncoder(TargetEncoder):
    """
    The slimmed TargetEncoder with 'train' and 'train_encode' attribute were set to None.
    """

    def __init__(self, n_folds=4, smooth=0, seed=42, split_method='interleaved', dtype=None, output_2d=False):
        super().__init__(n_folds=n_folds, smooth=smooth, seed=seed, split_method=split_method)

        self.dtype = dtype
        self.output_2d = output_2d

    def fit(self, X, y):
        super().fit(X, y)
        self.train = None
        self.train_encode = None
        return self

    def fit_transform(self, X, y):
        Xt, _ = self._fit_transform(X, y)
        self.train = None
        self.train_encode = None
        self._fitted = True
        if self.dtype is not None:
            Xt = Xt.astype(self.dtype)
        if self.output_2d:
            Xt = Xt.reshape(-1, 1)
        return Xt

    def transform(self, X):
        Xt = super().transform(X)
        if self.dtype is not None:
            Xt = Xt.astype(self.dtype)
        if self.output_2d:
            Xt = Xt.reshape(-1, 1)
        return Xt

    def _check_is_fitted(self):
        check_is_fitted(self, '_fitted')

    def _is_train_df(self, df):
        return False

    @property
    def split_method(self):
        return self.split


class ColumnEncoder(BaseEstimator):
    """
    Encode each column in the dataset with a separate encoder.
    """

    def create_encoder(self, X, y):
        raise NotImplementedError()

    def _check_X(self, X):
        assert len(X.shape) == 2

        if getattr(self, 'encoders_', None) is not None:  # fitted
            encoders = self.encoders_
            if self._is_dataframe(X):
                assert set(X.columns.tolist()) == set(encoders.keys())
            else:
                assert X.shape[1] == len(self.encoders_) \
                       and all([isinstance(k, int) for k in encoders.keys()])

    def _check_y(self, y):
        pass

    @staticmethod
    def _copy_X(X):
        return X.copy()

    @staticmethod
    def _is_dataframe(X):
        return hasattr(X, 'columns')

    def _call_fit_transform(self, encoder, Xc, y, **kwargs):
        if not hasattr(encoder, 'fit_transform'):
            self._call_fit(encoder, Xc, y, **kwargs)
            return self._call_transform(encoder, Xc)

        params = list(inspect.signature(encoder.fit_transform).parameters.values())
        if len(params) > 1 and params[1].kind in (params[1].POSITIONAL_ONLY, params[1].POSITIONAL_OR_KEYWORD):
            return encoder.fit_transform(Xc, y, **kwargs)
        else:
            return encoder.fit_transform(Xc, **kwargs)

    def _call_fit(self, encoder, Xc, y, **kwargs):
        params = list(inspect.signature(encoder.fit).parameters.values())
        if len(params) > 1 and params[1].kind in (params[1].POSITIONAL_ONLY, params[1].POSITIONAL_OR_KEYWORD):
            return encoder.fit(Xc, y, **kwargs)
        else:
            return encoder.fit(Xc, **kwargs)

    def _call_transform(self, encoder, Xc):
        return encoder.transform(Xc)

    def fit(self, X, y=None, **kwargs):
        self._check_X(X)
        self._check_y(y)

        columns = X.columns.tolist() if self._is_dataframe(X) else list(range(X.shape[1]))
        encoders = {c: self.create_encoder(X, y) for c in columns}

        for c, le in encoders.items():
            Xc = X[c] if self._is_dataframe(X) else X[:, c]
            self._call_fit(le, Xc, y, **kwargs)

        self.encoders_ = encoders

        return self

    def transform(self, X, *, copy=True):
        check_is_fitted(self, 'encoders_')
        self._check_X(X)
        if copy:
            X = self._copy_X(X)

        if self._is_dataframe(X):
            for c, le in self.encoders_.items():
                X[c] = le.transform(X[c])
        else:
            for c, le in self.encoders_.items():
                X[:, c] = le.transform(X[:, c])

        return X

    def fit_transform(self, X, y=None, *, copy=True, **kwargs):
        self._check_X(X)
        self._check_y(y)
        if copy:
            X = self._copy_X(X)

        columns = X.columns.tolist() if self._is_dataframe(X) else list(range(X.shape[1]))
        encoders = {c: self.create_encoder(X, y) for c in columns}

        if self._is_dataframe(X):
            for c, le in encoders.items():
                X[c] = self._call_fit_transform(le, X[c], y, **kwargs)
        else:
            for c, le in encoders.items():
                X[:, c] = self._call_fit_transform(le, X[:, c], y, **kwargs)

        self.encoders_ = encoders

        return X


@tb_transformer(pd.DataFrame)
class MultiTargetEncoder(ColumnEncoder):
    target_encoder_cls = SlimTargetEncoder
    label_encoder_cls = LabelEncoder

    def __init__(self, n_folds=4, smooth=None, seed=42, split_method='interleaved', dtype=None):
        self.n_folds = n_folds
        self.smooth = smooth
        self.seed = seed
        self.split_method = split_method
        self.dtype = dtype

    def create_encoder(self, X, y):
        smooth = int(len(X) ** .25) if self.smooth is None else self.smooth
        encoder = self.target_encoder_cls(n_folds=self.n_folds, smooth=smooth, seed=self.seed,
                                          split_method=self.split_method, dtype=self.dtype, output_2d=False)
        return encoder

    def fit(self, X, y=None, **kwargs):
        assert y is not None

        if str(y.dtype) == 'object':
            le = self.label_encoder_cls()
            y = le.fit_transform(y)
        return super().fit(X, y, **kwargs)

    def fit_transform(self, X, y=None, **kwargs):
        assert y is not None

        if str(y.dtype) == 'object':
            le = self.label_encoder_cls()
            y = le.fit_transform(y)
        return super().fit_transform(X, y, **kwargs)


@tb_transformer(pd.DataFrame)
class FeatureImportanceSelection(BaseEstimator):

    def __init__(self, importances, quantile, min_features=3):
        super(FeatureImportanceSelection, self).__init__()
        self.quantile = quantile
        self.importances = importances
        self.min_features = min_features

        n_features = int(round(len(self.importances) * (1 - self.quantile), 0))
        if n_features < min_features:
            n_features = min_features
        imps = [_[1] for _ in importances]
        self._important_features = [self.importances[i] for i in np.argsort(-np.array(imps))[: n_features]]

    def feature_usage(self):
        return len(self.important_features) / len(self.importances)

    def fit(self, X, y=None, **kwargs):
        return self

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)

    def transform(self, X):
        important_feature_names = [_[0] for _ in self.important_features]
        reversed_features = list(filter(lambda f: f in important_feature_names, X.columns.values))
        return X[reversed_features]

    @property
    def important_features(self):
        return self._important_features
