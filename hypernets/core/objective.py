import abc
from hypernets.core.searcher import OptimizeDirection


class Objective(metaclass=abc.ABCMeta):
    """ Objective  = Indicator metric + Direction"""

    def __init__(self, name, direction, need_train_data=False, need_val_data=True, need_test_data=False):
        self.name = name
        self.direction = direction
        self.need_train_data = need_train_data
        self.need_val_data = need_val_data
        self.need_test_data = need_test_data

    def evaluate(self, trial, estimator, X_train, y_train, X_val, y_val, X_test=None, **kwargs) -> float:
        if self.need_test_data:
            assert X_test is not None, "need test data"

        if self.need_train_data:
            assert X_train is not None and y_train is not None, "need train data"

        if self.need_val_data:
            assert X_val is not None and X_val is not None, "need validation data"

        return self._evaluate(trial, estimator, X_train, y_train, X_val, y_val, X_test=X_test, **kwargs)

    @abc.abstractmethod
    def _evaluate(self, trial, estimator, X_train, y_train, X_val, y_val, X_test=None, **kwargs) -> float:
        raise NotImplementedError

    def evaluate_cv(self, trial, estimator, X_trains, y_trains,
                    X_vals, y_vals, X_test=None, **kwargs) -> float:

        if self.need_test_data:
            assert X_test is not None, "need test data"

        if self.need_train_data:
            assert X_trains is not None and y_trains is not None, "need train data"
            assert len(X_trains) == len(y_trains)

        if self.need_val_data:
            assert X_vals is not None and y_vals is not None, "need validation data"
            assert len(X_vals) == len(y_vals)

        return self._evaluate_cv(trial=trial, estimator=estimator, X_trains=X_trains, y_trains=y_trains,
                                 X_vals=X_vals, y_vals=y_vals, X_test=X_test, **kwargs)

    @abc.abstractmethod
    def _evaluate_cv(self, trial, estimator, X_trains, y_trains, X_vals, y_vals, X_test=None, **kwargs) -> float:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, direction={self.direction}," \
               f" need_train_data={self.need_train_data}," \
               f" need_val_data={self.need_val_data}," \
               f" need_test_data={self.need_test_data})"
