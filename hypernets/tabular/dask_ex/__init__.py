# -*- coding:utf-8 -*-
"""

"""
import math
from functools import partial

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn import inspection as sk_inspect, metrics as sk_metrics
from sklearn import model_selection as sk_sel, utils as sk_utils
from sklearn.utils.multiclass import type_of_target

from hypernets.utils import logging

try:
    import dask_ml.preprocessing as dm_pre
    import dask_ml.model_selection as dm_sel

    from dask_ml.impute import SimpleImputer
    from dask_ml.compose import ColumnTransformer
    from dask_ml.preprocessing import \
        LabelEncoder, OneHotEncoder, OrdinalEncoder, \
        StandardScaler, MinMaxScaler, RobustScaler

    from ._transformers import \
        MultiLabelEncoder, SafeOneHotEncoder, TruncatedSVD, \
        MaxAbsScaler, SafeOrdinalEncoder, DataInterceptEncoder, \
        CallableAdapterEncoder, DataCacher, CacheCleaner, \
        LgbmLeavesEncoder, CategorizeEncoder, MultiKBinsDiscretizer, \
        LocalizedTfidfVectorizer, \
        MultiVarLenFeatureEncoder, DataFrameWrapper

    from ..sklearn_ex import PassThroughEstimator

    dask_ml_available = True
except ImportError:
    # Not found dask_ml
    dask_ml_available = False

logger = logging.get_logger(__name__)

compute = dask.compute


def default_client():
    try:
        from dask.distributed import default_client as dask_default_client
        client = dask_default_client()
    except ValueError:
        client = None
    return client


def dask_enabled():
    return default_client() is not None


def is_local_dask():
    client = default_client()
    return type(client.cluster).__name__.lower().find('local') >= 0 if client is not None else False


def dask_worker_count():
    client = default_client()
    return len(client.ncores()) if client else 0


def is_dask_dataframe(X):
    return isinstance(X, dd.DataFrame)


def is_dask_series(X):
    return isinstance(X, dd.Series)


def is_dask_dataframe_or_series(X):
    return isinstance(X, (dd.DataFrame, dd.Series))


def is_dask_array(X):
    return isinstance(X, da.Array)


def is_dask_object(X):
    return isinstance(X, (da.Array, dd.DataFrame, dd.Series))


def exist_dask_object(*args):
    for a in args:
        if isinstance(a, (da.Array, dd.DataFrame, dd.Series)):
            return True
        if isinstance(a, (tuple, list, set)):
            return exist_dask_object(*a)
    return False


def exist_dask_dataframe(*args):
    for a in args:
        if isinstance(a, dd.DataFrame):
            return True
        if isinstance(a, (tuple, list, set)):
            return exist_dask_dataframe(*a)
    return False


def exist_dask_array(*args):
    for a in args:
        if isinstance(a, da.Array):
            return True
        if isinstance(a, (tuple, list, set)):
            return exist_dask_array(*a)
    return False


def to_dask_type(X):
    if isinstance(X, np.ndarray):
        worker_count = dask_worker_count()
        chunk_size = math.ceil(X.shape[0] / worker_count) if worker_count > 0 else X.shape[0]
        X = da.from_array(X, chunks=chunk_size)
    elif isinstance(X, (pd.DataFrame, pd.Series)):
        worker_count = dask_worker_count()
        partition_count = worker_count if worker_count > 0 else 1
        X = dd.from_pandas(X, npartitions=partition_count)

    return X


def _reset_part_index(df, start):
    new_index = pd.RangeIndex.from_range(range(start, start + df.shape[0]))
    df.index = new_index
    return df


def reset_index(X):
    assert isinstance(X, (pd.DataFrame, dd.DataFrame))

    if is_dask_dataframe(X):
        part_rows = X.map_partitions(lambda df: pd.DataFrame({'rows': [df.shape[0]]}),
                                     meta={'rows': 'int64'},
                                     ).compute()['rows'].tolist()
        assert len(part_rows) == X.npartitions

        divisions = [0]
        n = 0
        for i in part_rows:
            n += i
            divisions.append(n)
        divisions[-1] = divisions[-1] - 1

        delayed_reset_part_index = dask.delayed(_reset_part_index)
        parts = [delayed_reset_part_index(part, start) for part, start in zip(X.to_delayed(), divisions[0:-1])]
        X_new = dd.from_delayed(parts, divisions=divisions, meta=X.dtypes.to_dict())
        return X_new
    else:
        return X.reset_index(drop=True)


def make_chunk_size_known(a):
    assert is_dask_array(a)

    chunks = a.chunks
    if any(np.nan in d for d in chunks):
        if logger.is_debug_enabled():
            logger.debug(f'call extracted array compute_chunk_sizes, shape: {a.shape}')
        a = a.compute_chunk_sizes()
    return a


def make_divisions_known(X):
    assert is_dask_object(X)

    if is_dask_dataframe(X):
        if not X.known_divisions:
            columns = X.columns.tolist()
            X = X.reset_index()
            new_columns = X.columns.tolist()
            index_name = set(new_columns) - set(columns)
            X = X.set_index(list(index_name)[0] if index_name else 'index')
            assert X.known_divisions
    elif is_dask_series(X):
        if not X.known_divisions:
            X = make_divisions_known(X.to_frame())[X.name]
    else:  # dask array
        X = make_chunk_size_known(X)

    return X


def hstack_array(arrs):
    if all([a.ndim == 1 for a in arrs]):
        rows = compute(arrs[0].shape)[0][0]
        arrs = [a.reshape(rows, 1) if a.ndim == 1 else a for a in arrs]
    return stack_array(arrs, axis=1)


def vstack_array(arrs):
    return stack_array(arrs, axis=0)


def stack_array(arrs, axis=0):
    assert axis in (0, 1)
    ndims = set([len(a.shape) for a in arrs])
    if len(ndims) > 1:
        assert ndims == {1, 2}
        assert all([len(a.shape) == 1 or a.shape[1] == 1 for a in arrs])
        arrs = [a.reshape(compute(a.shape[0])[0], 1) if len(a.shape) == 1 else a for a in arrs]
    axis = min(axis, min([len(a.shape) for a in arrs]) - 1)
    assert axis >= 0

    if exist_dask_object(*arrs):
        arrs = [a.values if is_dask_dataframe_or_series(a) else a for a in map(to_dask_type, arrs)]
        if len(arrs) > 1:
            arrs = [make_chunk_size_known(a) for a in arrs]
        return da.concatenate(arrs, axis=axis)
    else:
        return np.concatenate(arrs, axis=axis)


def array_to_df(arrs, columns=None, meta=None):
    meta_df = None
    if isinstance(meta, (dd.DataFrame, pd.DataFrame)):
        meta_df = meta
        if columns is None:
            columns = meta_df.columns
        # meta = dd.utils.make_meta(meta_df.dtypes.to_dict())
        if isinstance(meta, dd.DataFrame):
            meta = meta.head(0)
    elif isinstance(meta, (dd.Series, pd.Series)):
        meta_df = meta
        if columns is None:
            columns = meta_df.name
        meta = None

    df = dd.from_dask_array(arrs, columns=columns, meta=meta)

    if isinstance(meta_df, (dd.DataFrame, pd.DataFrame)):
        dtypes_src = meta_df.dtypes
        dtypes_dst = df.dtypes
        for col in meta_df.columns:
            if dtypes_src[col] != dtypes_dst[col]:
                df[col] = df[col].astype(dtypes_src[col])

    return df


def concat_df(dfs, axis=0, repartition=False, **kwargs):
    if exist_dask_object(*dfs):
        dfs = [dd.from_dask_array(v) if is_dask_array(v) else v for v in dfs]

        if all([isinstance(df, (dd.Series, pd.Series)) for df in dfs]):
            values = vstack_array([df.values for df in dfs])
            df = dd.from_dask_array(values, columns=dfs[0].name)
            assert isinstance(df, dd.Series)
            return df

        if axis == 0:
            values = [df[dfs[0].columns].to_dask_array(lengths=True) for df in dfs]
            df = array_to_df(vstack_array(values), meta=dfs[0])
        else:
            dfs = [make_divisions_known(df) for df in dfs]
            df = dd.concat(dfs, axis=axis, **kwargs)

        if is_dask_series(dfs[0]) and df.name is None and dfs[0].name is not None:
            df.name = dfs[0].name
        if repartition:
            df = df.repartition(npartitions=dfs[0].npartitions)
    else:
        df = pd.concat(dfs, axis=axis, **kwargs)

    return df


def train_test_split(*data, shuffle=True, random_state=None, **kwargs):
    if exist_dask_dataframe(*data):
        if len(data) > 1:
            data = [make_divisions_known(to_dask_type(x)) for x in data]
            head = data[0]
            for i in range(1, len(data)):
                if data[i].divisions != head.divisions:
                    print('-' * 10, f'repartition {i} from {data[i].divisions} to {head.divisions}')
                    data[i] = data[i].repartition(divisions=head.divisions)

        result = dm_sel.train_test_split(*data, shuffle=shuffle, random_state=random_state, **kwargs)
    else:
        result = sk_sel.train_test_split(*data, shuffle=shuffle, random_state=random_state, **kwargs)

    return result


def fix_binary_predict_proba_result(proba):
    if is_dask_object(proba):
        if proba.ndim == 1:
            proba = make_chunk_size_known(proba)
            proba = proba.reshape((proba.size, 1))
        if proba.shape[1] == 1:
            proba = hstack_array([1 - proba, proba])
    else:
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T
        elif proba.shape[1] == 1:
            proba = np.hstack([1 - proba, proba])

    return proba


def wrap_for_local_scorer(estimator, target_type):
    def _call_and_compute(fn_call, fn_fix, *args, **kwargs):
        r = fn_call(*args, **kwargs)
        if is_dask_object(r):
            r = r.compute()
            if callable(fn_fix):
                r = fn_fix(r)
        return r

    if hasattr(estimator, 'predict_proba'):
        orig_predict_proba = estimator.predict_proba
        fix = fix_binary_predict_proba_result if target_type == 'binary' else None
        setattr(estimator, '_orig_predict_proba', orig_predict_proba)
        setattr(estimator, 'predict_proba', partial(_call_and_compute, orig_predict_proba, fix))

    if hasattr(estimator, 'predict'):
        orig_predict = estimator.predict
        setattr(estimator, '_orig_predict', orig_predict)
        setattr(estimator, 'predict', partial(_call_and_compute, orig_predict, None))

    return estimator


def compute_and_call(fn_call, *args, **kwargs):
    if logger.is_debug_enabled():
        logger.debug(f'[compute_and_call] compute {len(args)} object')

    args = compute(*args, traverse=False)
    for k, v in kwargs.items():
        if exist_dask_object(v):
            kwargs[k] = compute(v, traverse=True)[0]

    if logger.is_debug_enabled():
        logger.debug(f'[compute_and_call] call {fn_call.__name__}')
    # kwargs = {k: compute(v) if is_dask_array(v) else v for k, v in kwargs.items()}
    r = fn_call(*args, **kwargs)

    if logger.is_debug_enabled():
        logger.debug('[compute_and_call] to dask type')
    r = to_dask_type(r)

    if logger.is_debug_enabled():
        logger.debug('[compute_and_call] done')
    return r


def call_and_compute(fn_call, optimize_graph, *args, **kwargs):
    if logger.is_debug_enabled():
        logger.debug(f'[call_and_compute] call {fn_call.__name__}')
    r = fn_call(*args, **kwargs)

    if is_dask_object(r):
        if logger.is_debug_enabled():
            logger.debug('[call_and_compute] to local type')
        r = compute(r, traverse=False)[0]
    elif isinstance(r, (tuple, list)) and any(map(is_dask_object, r)):
        if logger.is_debug_enabled():
            logger.debug('[call_and_compute] to local type')
        # r = compute(*r, traverse=False, optimize_graph=optimize_graph)
        r = [x.compute() if is_dask_object(x) else x for x in r]

    if logger.is_debug_enabled():
        logger.debug('[call_and_compute] done')

    return r


def wrap_local_estimator(estimator):
    for fn_name in ('fit', 'fit_cross_validation', 'predict', 'predict_proba'):
        fn_name_original = f'_wrapped_{fn_name}_by_wle'
        if hasattr(estimator, fn_name) and not hasattr(estimator, fn_name_original):
            fn = getattr(estimator, fn_name)
            assert callable(fn)
            setattr(estimator, fn_name_original, fn)
            setattr(estimator, fn_name, partial(compute_and_call, fn))

    return estimator


def permutation_importance(estimator, X, y, *args, scoring=None, n_repeats=5,
                           n_jobs=None, random_state=None):
    if not is_dask_dataframe(X):
        return sk_inspect.permutation_importance(estimator, X, y, *args,
                                                 scoring=scoring,
                                                 n_repeats=n_repeats,
                                                 n_jobs=n_jobs,
                                                 random_state=random_state)
    random_state = sk_utils.check_random_state(random_state)

    def shuffle_partition(df, col_idx):
        shuffling_idx = np.arange(df.shape[0])
        random_state.shuffle(shuffling_idx)
        col = df.iloc[shuffling_idx, col_idx]
        col.index = df.index
        df.iloc[:, col_idx] = col
        return df

    if is_dask_object(y):
        y = y.compute()

    scorer = sk_metrics.check_scoring(wrap_for_local_scorer(estimator, type_of_target(y)), scoring)
    baseline_score = scorer(estimator, X, y)
    scores = []

    for c in range(X.shape[1]):
        col_scores = []
        for i in range(n_repeats):
            X_permuted = X.copy().map_partitions(shuffle_partition, c)
            col_scores.append(scorer(estimator, X_permuted, y))
        if logger.is_debug_enabled():
            logger.debug(f'permuted scores [{X.columns[c]}]: {col_scores}')
        scores.append(col_scores)

    importances = baseline_score - np.array(scores)
    return sk_utils.Bunch(importances_mean=np.mean(importances, axis=1),
                          importances_std=np.std(importances, axis=1),
                          importances=importances)


@sk_utils._deprecate_positional_args
def compute_class_weight(class_weight, *, classes, y):
    # f"""{sk_utils.class_weight.compute_class_weight.__doc__}"""

    if not is_dask_object(y):
        return sk_utils.class_weight.compute_class_weight(class_weight, classes=classes, y=y)

    y = make_chunk_size_known(y)
    if set(compute(da.unique(y))[0]) - set(classes):
        raise ValueError("classes should include all valid labels that can be in y")

    if class_weight == 'balanced':
        # Find the weight of each class as present in y.
        le = dm_pre.LabelEncoder()
        y_ind = le.fit_transform(y)
        # if not all(np.in1d(classes, le.classes_)):
        #     raise ValueError("classes should have valid labels that are in y")
        # recip_freq = len(y) / (len(le.classes_) *
        #                        np.bincount(y_ind).astype(np.float64))
        # weight = recip_freq[le.transform(classes)]
        y_shape, y_ind_bincount, le_classes_ = compute(y.shape, da.bincount(y_ind), le.classes_)
        if not all(np.in1d(classes, le_classes_)):
            raise ValueError("classes should have valid labels that are in y")
        recip_freq = y_shape[0] / (len(le_classes_) * y_ind_bincount.astype(np.float64))
        weight = recip_freq[np.searchsorted(le_classes_, classes)]
    else:
        raise ValueError("Only class_weight == 'balanced' is supported.")

    return weight


def _compute_chunk_sample_weight(y, classes, classes_weights):
    t = np.ones(y.shape[0])
    for i, c in enumerate(classes):
        t[y == c] *= classes_weights[i]
    return t


def compute_sample_weight(y):
    assert len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1)

    if is_dask_dataframe_or_series(y):
        y = y.values

    unique = compute(da.unique(y))[0] if is_dask_object(y) else np.unique(y)
    cw = list(compute_class_weight('balanced', classes=unique, y=y))

    if is_dask_object(y):
        sample_weight = y.map_blocks(_compute_chunk_sample_weight, unique, cw, dtype=np.float64)
    else:
        sample_weight = _compute_chunk_sample_weight(y, unique, cw)

    return sample_weight
