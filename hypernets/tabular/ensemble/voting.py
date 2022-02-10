# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from collections import defaultdict

import joblib
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _PredictScorer

from .base_ensemble import BaseEnsemble
from ..cfg import TabularCfg as cfg


class AveragingEnsemble(BaseEnsemble):
    def __init__(self, task, estimators, need_fit=False, n_folds=5, method='soft'):
        super(AveragingEnsemble, self).__init__(task, estimators, need_fit, n_folds, method)

    def fit_predictions(self, predictions, y_true):
        return self

    def predictions2predict(self, predictions):
        if len(predictions.shape) == 3 and self.task == 'binary':
            predictions = predictions[:, :, -1]
        np = self.np
        proba = np.mean(predictions, axis=1)
        pred = self.proba2predict(proba)
        return pred

    def predictions2predict_proba(self, predictions):
        if self.task == 'multiclass' and self.method == 'hard':
            raise ValueError('Multiclass task does not support `hard` method.')
        np = self.np
        proba = np.mean(predictions, axis=1)
        if self.task == 'regression':
            return proba
        proba = np.clip(proba, 0, 1)
        if len(proba.shape) == 1:
            proba = np.stack([1 - proba, proba], axis=1)
        return proba


class GreedyEnsemble(BaseEnsemble):
    """
    References
    ----------
        Caruana, Rich, et al. "Ensemble selection from libraries of models." Proceedings of the twenty-first international conference on Machine learning. 2004.
    """

    def __init__(self, task, estimators, need_fit=False, n_folds=5, method='soft', random_state=9527,
                 scoring='neg_log_loss', ensemble_size=0):
        super(GreedyEnsemble, self).__init__(task, estimators, need_fit, n_folds, method, random_state=random_state)
        self.scoring = scoring
        self.scorer = get_scorer(scoring)
        self.ensemble_size = ensemble_size

        # fitted
        self.weights_ = None
        self.scores_ = None
        self.best_stack_ = None
        self.hits_ = None

    def __repr__(self) -> str:
        if self.estimators is None:
            return 'no estimators'

        if self.weights_ is None:
            return 'not fitted'

        # estimators = [getattr(e, "gbm_model", e) for e in self.estimators]
        return f'{type(self).__name__}(weight={self.weights_}, scores={self.scores_})'

    def _repr_html_(self):
        import pandas as pd
        df = pd.DataFrame([('weights', self.weights_),
                           ('scores', self.scores_),
                           ('best_stack', self.best_stack_),
                           ('hits', self.hits_),
                           ('ensemble_size', self.ensemble_size)])
        return df._repr_html_()

    # def _score(self, y_true, y_pred):
    #     return self.scorer._score_func(y_true, y_pred, **self.scorer._kwargs) * self.scorer._sign

    def _score(self, y_ture, y_preds):
        fn = joblib.delayed(self.scorer._score_func)
        paral = joblib.Parallel(n_jobs=cfg.joblib_njobs, **cfg.joblib_options)
        rs = paral(fn(y_ture, p, **self.scorer._kwargs) for p in y_preds)
        rs = [r * self.scorer._sign for r in rs]
        return rs

    def fit_predictions(self, predictions, y_true):
        np = self.np
        scores = []
        best_stack = []
        if len(predictions.shape) == 1:
            self.weights_ = [1]
            return
        elif len(predictions.shape) == 2:
            sum_predictions = np.zeros((predictions.shape[0]), dtype=np.float64)
        elif len(predictions.shape) == 3:
            sum_predictions = np.zeros((predictions.shape[0], predictions.shape[2]), dtype=np.float64)
        else:
            raise ValueError(f'Wrong shape of predictions. shape:{predictions.shape}')

        if self.ensemble_size <= 0:
            size = predictions.shape[1]
        else:
            size = self.ensemble_size
        for i in range(size):
            # stack_scores = []
            preds = []
            for j in range(predictions.shape[1]):
                if len(predictions.shape) == 2:
                    pred = predictions[:, j]
                else:
                    pred = predictions[:, j, :]
                mean_predictions = (sum_predictions + pred) / (len(best_stack) + 1)
                if isinstance(self.scorer, _PredictScorer) and self.classes_ is not None and len(self.classes_) > 0:
                    # pred = np.take(np.array(self.classes_), np.argmax(mean_predictions, axis=1), axis=0)
                    pred = self._indices2predict(np.argmax(mean_predictions, axis=1))
                    mean_predictions = pred
                elif self.task == 'binary' and len(mean_predictions.shape) == 2 and mean_predictions.shape[1] == 2:
                    mean_predictions = mean_predictions[:, 1]
                preds.append(mean_predictions)
                # score = self._score(y_true, mean_predictions)
                # stack_scores.append(score)
            stack_scores = self._score(y_true, preds)

            # best = np.argmax(stack_scores)
            # scores.append(stack_scores[best])
            best, best_score = (0, stack_scores[0])
            for n, score in enumerate(stack_scores):
                if score > best_score:
                    best, best_score = (n, score)
            scores.append(best_score)

            best_stack.append(best)
            if len(predictions.shape) == 2:
                sum_predictions += predictions[:, best]
            else:
                sum_predictions += predictions[:, best, :]

        # best_step = int(np.argmax(scores))
        # print(f'best_step:{best_step}')
        # val_steps = best_step + 1

        # sum up estimator's hit count
        val_steps = len(best_stack)
        hits = defaultdict(int)
        for i in range(val_steps):
            hits[best_stack[i]] += 1

        weights = np.zeros((len(self.estimators)), dtype=np.float64)
        for i in range(len(self.estimators)):
            if hits.get(i) is not None:
                weights[i] = hits[i] / val_steps

        # zero_weight_index = np.argwhere(weights == 0.).ravel()
        # for index in zero_weight_index:
        #     self.estimators[index] = None
        for index, weight in enumerate(weights):
            if weight == 0.0:
                self.estimators[index] = None

        self.weights_ = weights.tolist()
        self.scores_ = scores
        self.hits_ = hits
        self.best_stack_ = best_stack

    def predictions2predict(self, predictions):
        assert len(self.weights_) == predictions.shape[1]
        np = self.np
        weights = np.array(self.weights_)
        if len(predictions.shape) == 3 and self.task == 'binary':
            predictions = predictions[:, :, -1]
        if len(predictions.shape) == 3:
            weights = np.expand_dims(weights, axis=1).repeat(predictions.shape[2], 1)

        proba = np.sum(predictions * weights, axis=1)
        pred = self.proba2predict(proba)
        return pred

    def predictions2predict_proba(self, predictions):
        assert len(self.weights_) == predictions.shape[1]
        if self.task == 'multiclass' and self.method == 'hard':
            raise ValueError('Multiclass task does not support `hard` method.')
        np = self.np
        weights = np.array(self.weights_)
        if len(predictions.shape) == 3:
            weights = np.expand_dims(weights, axis=1).repeat(predictions.shape[2], 1)

        proba = np.sum(predictions * weights, axis=1)

        if self.task == 'regression':
            return proba
        else:
            # guaranteed to sum to 1.0 over classes
            proba = proba * np.expand_dims(1 / (proba.sum(axis=1)), axis=1).repeat(proba.shape[1], 1)

        if len(proba.shape) == 1:
            proba = np.stack([1 - proba, proba], axis=1)
        return proba
