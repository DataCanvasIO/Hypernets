from hypernets.tabular import sklearn_ex as skex, get_tool_box
from hypernets.tabular.cache import cache, clear, CacheCallback
from hypernets.tabular.datasets import dsutils
from hypernets.utils import Counter


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


def test_cache():
    clear()

    df = dsutils.load_bank()
    t = skex.MultiLabelEncoder()
    X = t.fit_transform(df.copy())

    t1 = CachedMultiLabelEncoder()
    X1 = t1.fit_transform(df.copy())
    t2 = CachedMultiLabelEncoder()
    X2 = t2.fit_transform(df.copy())

    hasher = get_tool_box(df).data_hasher()
    assert hasher(X) == hasher(X1) == hasher(X2)

    t3 = CachedMultiLabelEncoder()
    X3 = t3.fit_transform_as_tuple_result(df.copy())
    t4 = CachedMultiLabelEncoder()
    X4 = t4.fit_transform_as_tuple_result(df.copy())
    assert isinstance(X3, (tuple, list))
    assert isinstance(X4, (tuple, list))
    assert hasher(X3[1]) == hasher(X4[1])
