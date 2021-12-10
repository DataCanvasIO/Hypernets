# -*- coding:utf-8 -*-
"""

"""
import pytest

from hypernets.tabular.datasets.dsutils import load_bank
from . import if_cuml_ready, is_cuml_installed

if is_cuml_installed:
    import cudf
    import cupy
    from cuml.model_selection import train_test_split as split
    from hypernets.tabular.cuml_ex._model_selection import train_test_split as tplit


def load_data(y='y'):
    df = cudf.from_pandas(load_bank().head(1000))
    X = df
    y = X.pop(y)
    return X, y


@if_cuml_ready
class TestCumlDataSplit:
    def test_y_int(self):
        X, y = load_data('day')
        assert str(y.dtype).find('int') >= 0
        X_train, X_test, y_train, y_test = split(X, y, shuffle=True, random_state=1)

    def test_y_int_stratify(self):
        X, y = load_data('day')
        assert str(y.dtype).find('int') >= 0
        X_train, X_test, y_train, y_test = split(X, y, test_size=0.5, shuffle=True, random_state=1, stratify=y)

    def test_y_float(self):
        X, y = load_data('day')
        y = y.astype('float32')
        assert str(y.dtype) == 'float32'
        X_train, X_test, y_train, y_test = split(X, y, shuffle=True, random_state=1)

    @pytest.mark.xfail('NotImplemented')
    def test_y_object(self):
        X, y = load_data('y')
        assert str(y.dtype) == 'object'
        X_train, X_test, y_train, y_test = split(X, y, shuffle=True, random_state=1)

    @pytest.mark.xfail('NotImplemented')
    def test_y_object_stratify(self):
        X, y = load_data('y')
        assert str(y.dtype) == 'object'
        X_train, X_test, y_train, y_test = split(X, y, shuffle=True, random_state=1, stratify=y)


@if_cuml_ready
class TestCumlToolboxDataSplit:
    def test_y_int(self):
        X, y = load_data('day')
        assert str(y.dtype).find('int') >= 0
        X_train, X_test, y_train, y_test = tplit(X, y, shuffle=True, random_state=1)

    def test_y_int_stratify(self):
        X, y = load_data('day')
        assert str(y.dtype).find('int') >= 0
        X_train, X_test, y_train, y_test = tplit(X, y, test_size=0.5, shuffle=True, random_state=1, stratify=y)

    def test_y_float(self):
        X, y = load_data('day')
        y = y.astype('float32')
        assert str(y.dtype) == 'float32'
        X_train, X_test, y_train, y_test = tplit(X, y, shuffle=True, random_state=1)

    def test_y_object(self):
        X, y = load_data('y')
        assert str(y.dtype) == 'object'
        X_train, X_test, y_train, y_test = tplit(X, y, shuffle=True, random_state=1)
        assert str(y_train.dtype) == 'object'

    def test_y_object_stratify(self):
        X, y = load_data('y')
        assert str(y.dtype) == 'object'
        X_train, X_test, y_train, y_test = tplit(X, y, shuffle=True, random_state=1, stratify=y)
        assert str(y_train.dtype) == 'object'

    def test_result(self):
        df = cudf.DataFrame(dict(
            x1=[1, 2, 3, 4, 5],
            x2=['a', 'b', 'c', 'd', 'e'],
            y=['x', 'y', 'z', 'x', 'y']
        ))
        X = df.copy()
        y = X.pop('y')

        X_train, X_test, y_train, y_test = tplit(X, y, test_size=0.5, shuffle=True, random_state=1)
        assert cupy.asnumpy(X_train.index.values == y_train.index.values).all()
        assert cupy.asnumpy(X_test.index.values == y_test.index.values).all()

        X_train['y'] = y_train
        X_test['y'] = y_test
        tf = cudf.concat([X_train, X_test]).sort_index()
        assert (df == tf).all(skipna=False).all()
        print(tf)
