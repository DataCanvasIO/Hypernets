import abc
from hypernets.core.searcher import OptimizeDirection


class Objective(metaclass=abc.ABCMeta):
    """ Objective  = Indicator metric + Direction
    """

    def __init__(self, name, direction):
        self.name = name
        self.direction = direction

    @abc.abstractmethod
    def call(self, trial, estimator, X_test, y_test, **kwargs) -> float:
        raise NotImplementedError

    def call_cross_validation(self, trial, estimators, X_tests, y_tests, **kwargs) -> float:
        assert len(estimators) == len(X_tests) == len(y_tests)

        return self._call_cross_validation(trial=trial, estimators=estimators, X_tests=X_tests,
                                           y_tests=y_tests, **kwargs)

    @abc.abstractmethod
    def _call_cross_validation(self, trial, estimators, X_tests, y_tests, **kwargs) -> float:
        raise NotImplementedError

    def __call__(self, trial, estimator, X_test, y_test, **kwargs):
        return self.call(trial=trial, estimator=estimator, X_test=X_test, y_test=y_test, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, direction={self.direction})"
