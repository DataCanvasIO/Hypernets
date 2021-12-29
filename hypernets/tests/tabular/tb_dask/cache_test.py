from hypernets.tabular.cache import cache, clear
from hypernets.tabular.datasets import dsutils
from . import if_dask_ready, is_dask_installed
from ..cache_test import CacheCounter

if is_dask_installed:
    import dask.dataframe as dd
    from hypernets.tabular import dask_ex as dex


    class CachedDaskMultiLabelEncoder(dex.SafeOrdinalEncoder):
        cache_counter = CacheCounter()

        @cache(attr_keys='columns',
               attrs_to_restore='columns,dtype,categorical_columns_,non_categorical_columns_,categories_',
               callbacks=cache_counter)
        def fit_transform(self, X, *args):
            return super().fit_transform(X, *args)

        @cache(attr_keys='columns',
               attrs_to_restore='columns,dtype,categorical_columns_,non_categorical_columns_,categories_',
               callbacks=cache_counter)
        def fit_transform_as_array(self, X, *args):
            X = super().fit_transform(X, *args)
            return X.to_dask_array(lengths=True)


@if_dask_ready
def test_cache_dask():
    clear()

    cache_counter = CachedDaskMultiLabelEncoder.cache_counter
    df = dd.from_pandas(dsutils.load_bank(), npartitions=2)

    t = dex.SafeOrdinalEncoder()
    X = t.fit_transform(df.copy())

    cache_counter.reset()
    t1 = CachedDaskMultiLabelEncoder()
    X1 = t1.fit_transform(df.copy())
    t2 = CachedDaskMultiLabelEncoder()
    X2 = t2.fit_transform(df.copy())

    hasher = dex.DaskToolBox.data_hasher()
    assert hasher(X) == hasher(X1) == hasher(X2)
    assert cache_counter.enter_counter.value == 2
    assert cache_counter.apply_counter.value <= 2
    assert cache_counter.store_counter.value <= 2
    assert cache_counter.apply_counter.value + cache_counter.store_counter.value == 2

    cache_counter.reset()
    t3 = CachedDaskMultiLabelEncoder()
    X3 = t3.fit_transform_as_array(df.copy())
    t4 = CachedDaskMultiLabelEncoder()
    X4 = t4.fit_transform_as_array(df.copy())

    assert hasher(X3) == hasher(X4)
    assert cache_counter.enter_counter.value == 2
    assert cache_counter.apply_counter.value <= 2
    assert cache_counter.store_counter.value <= 2
    assert cache_counter.apply_counter.value + cache_counter.store_counter.value == 2
