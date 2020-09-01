# -*- coding:utf-8 -*-
"""

"""
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_log_error, accuracy_score, \
    mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score, log_loss


# class DtypeCastTransformer(TransformerMixin, BaseEstimator):
#     def __init__(self):


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
        for n in range(n_features):
            le = SafeLabelEncoder()
            le.fit(X[:, n])
            self.encoders[n] = le
        return self

    def transform(self, X):
        assert len(X.shape) == 2
        n_features = X.shape[1]
        assert n_features == len(self.encoders.items())
        for n in range(n_features):
            X[:, n] = self.encoders[n].transform(X[:, n])
        return X


def calc_score(y_true, y_preds, y_proba=None, metrics=['accuracy'], task='binary', pos_label=1):
    score = {}
    if y_proba is None:
        y_proba = y_preds
    if len(y_proba.shape) == 2 and y_proba.shape[-1] == 1:
        y_proba = y_proba.reshape(-1)
    if len(y_preds.shape) == 2 and y_preds.shape[-1] == 1:
        y_preds = y_preds.reshape(-1)
    for metric in metrics:
        if callable(metric):
            score[metric.__name__] = metric(y_true, y_preds)
        else:
            metric = metric.lower()
            if task == 'multiclass':
                average = 'micro'
            else:
                average = 'binary'

            if metric == 'auc':
                if len(y_proba.shape) == 2:

                    score['auc'] = roc_auc_score(y_true, y_proba[:, 1], multi_class='ovo')
                else:
                    score['auc'] = roc_auc_score(y_true, y_proba)

            elif metric == 'accuracy':
                if y_proba is None:
                    score['accuracy'] = 0
                else:
                    score['accuracy'] = accuracy_score(y_true, y_preds)
            elif metric == 'recall':
                score['recall'] = recall_score(y_true, y_preds, average=average, pos_label=pos_label)
            elif metric == 'precision':
                score['precision'] = precision_score(y_true, y_preds, average=average, pos_label=pos_label)
            elif metric == 'f1':
                score['f1'] = f1_score(y_true, y_preds, average=average, pos_label=pos_label)

            elif metric == 'mse':
                score['mse'] = mean_squared_error(y_true, y_preds)
            elif metric == 'mae':
                score['mae'] = mean_absolute_error(y_true, y_preds)
            elif metric == 'msle':
                score['msle'] = mean_squared_log_error(y_true, y_preds)
            elif metric == 'rmse':
                score['rmse'] = np.sqrt(mean_squared_error(y_true, y_preds))
            elif metric == 'rootmeansquarederror':
                score['rootmeansquarederror'] = np.sqrt(mean_squared_error(y_true, y_preds))
            elif metric == 'r2':
                score['r2'] = r2_score(y_true, y_preds)
            elif metric == 'logloss':
                score['logloss'] = log_loss(y_true, y_proba)

    return score
