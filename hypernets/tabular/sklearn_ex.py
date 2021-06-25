# -*- coding:utf-8 -*-
"""

"""
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
from hypernets.utils import logging, infer_task_type, const

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


class PassThroughEstimator(BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


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


class MultiLabelEncoder(BaseEstimator):
    def __init__(self, columns=None):
        super(MultiLabelEncoder, self).__init__()

        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        assert len(X.shape) == 2
        assert isinstance(X, pd.DataFrame) or self.columns is None

        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                self.columns = X.columns.tolist()
            for col in self.columns:
                data = X.loc[:, col]
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
                data = X.loc[:, col]
                if data.dtype == 'object':
                    data = data.astype('str')
                X.loc[:, col] = self.encoders[col].transform(data)
        else:
            n_features = X.shape[1]
            assert n_features == len(self.encoders.items())
            for n in range(n_features):
                X[:, n] = self.encoders[n].transform(X[:, n])

        return X

    def fit_transform(self, X, *args):
        assert len(X.shape) == 2
        assert isinstance(X, pd.DataFrame) or self.columns is None

        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                self.columns = X.columns.tolist()
            for col in self.columns:
                data = X.loc[:, col]
                if data.dtype == 'object':
                    data = data.astype('str')
                    # print(f'Column "{col}" has been convert to "str" type.')
                le = SafeLabelEncoder()
                X.loc[:, col] = le.fit_transform(data)
                self.encoders[col] = le
        else:
            n_features = X.shape[1]
            for n in range(n_features):
                data = X[:, n]
                le = SafeLabelEncoder()
                X[:, n] = le.fit_transform(data)
                self.encoders[n] = le

        return X


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


class FloatOutputImputer(SimpleImputer):

    def transform(self, X):
        return super().transform(X).astype(np.float64)


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
            n_unique = X.loc[:, col].nunique()
            n_null = X.loc[:, col].isnull().sum()
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

        from tensorflow.python.keras.preprocessing.sequence import pad_sequences

        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
        data = X.map(lambda _: (self.encoder.transform(_.split(self.sep)) + 1).tolist())

        transformed = pad_sequences(data, maxlen=self._max_element_length, padding='post',
                                    truncating='post').tolist()  # cut last elements
        return transformed

    @property
    def n_classes(self):
        return len(self.encoder.classes_)

    @property
    def max_element_length(self):
        return self._max_element_length


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

    def fit(self, X, y=None):
        assert isinstance(X, (np.ndarray, pd.DataFrame)) and len(X.shape) == 2

        if self.columns is None:
            if isinstance(X, pd.DataFrame):
                columns = column_selector.column_object(X)
            else:
                columns = range(X.shape[1])
        else:
            columns = self.columns

        encoders = {}
        for c in columns:
            encoder = LocalizedTfidfVectorizer(**self.encoder_kwargs)
            Xc = X[c] if isinstance(X, pd.DataFrame) else X[:, c]
            encoders[c] = encoder.fit(Xc, y)

        self.encoders_ = encoders

        return self

    def transform(self, X, y=None):
        assert self.encoders_ is not None
        assert isinstance(X, (np.ndarray, pd.DataFrame)) and len(X.shape) == 2

        if isinstance(X, pd.DataFrame):
            X = X.copy()
            if self.flatten:
                dfs = [X]
                for c, encoder in self.encoders_.items():
                    t = encoder.transform(X[c]).toarray()
                    dfs.append(pd.DataFrame(t, index=X.index, columns=[f'{c}_tfidf_{i}' for i in range(t.shape[1])]))
                    X.pop(c)
                X = pd.concat(dfs, axis=1)
            else:
                for c, encoder in self.encoders_.items():
                    t = encoder.transform(X[c]).toarray()
                    X[c] = t.tolist()
        else:
            r = []
            tolist = None if self.flatten else np.vectorize(self._to_array, otypes=[np.object], signature='(m)->()')
            for i in range(X.shape[1]):
                Xi = X[:, i]
                if i in self.encoders_.keys():
                    encoder = self.encoders_[i]
                    t = encoder.transform(Xi).toarray()
                    if tolist is not None:
                        t = tolist(t).reshape((-1, 1))
                    r.append(t)
                else:
                    r.append(Xi)
            X = np.hstack(r)

        return X

    @staticmethod
    def _to_list(x):
        return x.tolist()

    @staticmethod
    def _to_array(x):
        return x


class DatetimeEncoder(BaseEstimator, TransformerMixin):
    all_items = ['year', 'month', 'day', 'hour', 'minute', 'second',
                 'week', 'weekday', 'dayofyear',
                 'timestamp']
    all_items = {k: k for k in all_items}
    all_items['timestamp'] = lambda t: time.mktime(t.timetuple())

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
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            self.columns = column_selector.column_all_datetime(X)

        return self

    def transform(self, X, y=None):
        if len(self.columns) > 0:
            if isinstance(X, pd.DataFrame):
                X = X.copy()
                input_df = True
            else:
                X = pd.DataFrame(X)
                input_df = False
            dfs = [df for c in self.columns for df in self.transform_column(X[c])]
            X.drop(columns=self.columns, inplace=True)
            if len(dfs) > 0:
                dfs.insert(0, X)
                X = pd.concat(dfs, axis=1)
            if not input_df:
                X = X.values

        return X

    def transform_column(self, Xc):
        assert getattr(Xc, 'dt', None) is not None
        dfs = []

        for k, c in self.extract_.items():
            if c is None:
                c = k
            if isinstance(c, str):
                t = getattr(Xc.dt, c)
            else:
                t = Xc.apply(c)
            t.name = f'{Xc.name}_{k}'
            dfs.append(t)

        if self.drop_constants:
            dfs = [t for t in dfs if t.nunique() > 1]

        return dfs
