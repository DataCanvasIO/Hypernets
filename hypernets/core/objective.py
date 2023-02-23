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

    def __call__(self, trial, estimator, X_test, y_test, **kwargs):
        return self.call(trial=trial, estimator=estimator, X_test=X_test, y_test=y_test, **kwargs)
