from hypernets.tabular import sklearn_ex as skex, dask_ex as dex
from hypernets.tabular.cache import cache, CacheCallback
from hypernets.tabular.datasets import dsutils
from hypernets.utils import hash_data, Counter


class CacheCounter(CacheCallback):
    def __init__(self):
        super(CacheCounter, self).__init__()

        self.enter_counter = Counter()
        self.apply_counter = Counter()
        self.store_counter = Counter()

    def on_enter(self, fn, *args, **kwargs):
        self.enter_counter()

    def on_apply(self, fn, cached_data, *args, **kwargs):
        self.apply_counter()

    def on_store(self, fn, cached_data, *args, **kwargs):
        self.store_counter()

    def reset(self):
        self.enter_counter.reset()
        self.apply_counter.reset()
        self.store_counter.reset()


class CachedMultiLabelEncoder(skex.MultiLabelEncoder):
    @cache(attr_keys='columns', attrs_to_restore='columns,encoders')
    def fit_transform(self, X, *args):
        return super().fit_transform(X, *args)

    @cache(attr_keys='columns', attrs_to_restore='columns,encoders')
    def fit_transform_as_tuple_result(self, X, *args):
        Xt = super().fit_transform(X.copy(), *args)
        return X, Xt


class CachedDaskMultiLabelEncoder(dex.MultiLabelEncoder):
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


def test_cache():
    df = dsutils.load_bank().head(10000)
    t = skex.MultiLabelEncoder()
    X = t.fit_transform(df.copy())

    t1 = CachedMultiLabelEncoder()
    X1 = t1.fit_transform(df.copy())
    t2 = CachedMultiLabelEncoder()
    X2 = t2.fit_transform(df.copy())
    assert hash_data(X) == hash_data(X1) == hash_data(X2)

    t3 = CachedMultiLabelEncoder()
    X3 = t3.fit_transform_as_tuple_result(df.copy())
    t4 = CachedMultiLabelEncoder()
    X4 = t4.fit_transform_as_tuple_result(df.copy())
    assert isinstance(X3, (tuple, list))
    assert isinstance(X4, (tuple, list))
    assert hash_data(X3[1]) == hash_data(X4[1])


def test_cache_dask():
    cache_counter = CachedDaskMultiLabelEncoder.cache_counter
    df = dex.dd.from_pandas(dsutils.load_bank().head(10000), npartitions=2)

    t = dex.MultiLabelEncoder()
    X = t.fit_transform(df.copy())

    cache_counter.reset()
    t1 = CachedDaskMultiLabelEncoder()
    X1 = t1.fit_transform(df.copy())
    t2 = CachedDaskMultiLabelEncoder()
    X2 = t2.fit_transform(df.copy())

    assert hash_data(X) == hash_data(X1) == hash_data(X2)
    assert cache_counter.enter_counter.value == 2
    assert cache_counter.apply_counter.value <= 2
    assert cache_counter.store_counter.value <= 2
    assert cache_counter.apply_counter.value + cache_counter.store_counter.value == 2

    cache_counter.reset()
    t3 = CachedDaskMultiLabelEncoder()
    X3 = t3.fit_transform_as_array(df.copy())
    t4 = CachedDaskMultiLabelEncoder()
    X4 = t4.fit_transform_as_array(df.copy())

    assert hash_data(X3) == hash_data(X4)
    assert cache_counter.enter_counter.value == 2
    assert cache_counter.apply_counter.value <= 2
    assert cache_counter.store_counter.value <= 2
    assert cache_counter.apply_counter.value + cache_counter.store_counter.value == 2
