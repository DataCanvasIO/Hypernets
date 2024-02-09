try:
    from sklearn.metrics._scorer import _PredictScorer


    def is_predict_scorer(s):
        return isinstance(s, _PredictScorer)
except ImportError:
    # sklearn 1.4.0 +
    def is_predict_scorer(s):
        return getattr(s, '_response_method', '') == 'predict'
