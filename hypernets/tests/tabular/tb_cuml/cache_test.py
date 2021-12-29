from hypernets.tabular.cache import cache, clear
from hypernets.tabular.datasets import dsutils
from . import if_cuml_ready, is_cuml_installed
from ..cache_test import CacheCounter

if is_cuml_installed:
    import cudf
    from hypernets.tabular.cuml_ex import CumlToolBox
    from hypernets.tabular.cuml_ex._transformer import MultiLabelEncoder


    class CachedCumlMultiLabelEncoder(MultiLabelEncoder):
        cache_counter = CacheCounter()

        @cache(attr_keys='columns',
               attrs_to_restore='columns,dtype,encoders',
               callbacks=cache_counter)
        def fit_transform(self, X, *args):
            return super().fit_transform(X, *args)

        @cache(attr_keys='columns',
               attrs_to_restore='columns,dtype,encoders',
               callbacks=cache_counter)
        def fit_transform_as_array(self, X, *args):
            X = super().fit_transform(X, *args)
            return X.values


@if_cuml_ready
def test_cache_cuml():
    clear()

    cache_counter = CachedCumlMultiLabelEncoder.cache_counter
    df = cudf.from_pandas(dsutils.load_bank())

    t = MultiLabelEncoder()
    X = t.fit_transform(df.copy())

    cache_counter.reset()
    t1 = CachedCumlMultiLabelEncoder()
    X1 = t1.fit_transform(df.copy())
    t2 = CachedCumlMultiLabelEncoder()
    X2 = t2.fit_transform(df.copy())

    hasher = CumlToolBox.data_hasher()
    assert hasher(X) == hasher(X1) == hasher(X2)
    assert cache_counter.enter_counter.value == 2
    assert cache_counter.apply_counter.value <= 2
    assert cache_counter.store_counter.value <= 2
    assert cache_counter.apply_counter.value + cache_counter.store_counter.value == 2

    cache_counter.reset()
    t3 = CachedCumlMultiLabelEncoder()
    X3 = t3.fit_transform_as_array(df.copy())
    t4 = CachedCumlMultiLabelEncoder()
    X4 = t4.fit_transform_as_array(df.copy())

    assert hasher(X3) == hasher(X4)
    assert cache_counter.enter_counter.value == 2
    assert cache_counter.apply_counter.value <= 2
    assert cache_counter.store_counter.value <= 2
    assert cache_counter.apply_counter.value + cache_counter.store_counter.value == 2
