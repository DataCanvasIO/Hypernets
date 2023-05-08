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
from dask_ml import compose as dm_comp
from dask_ml import model_selection as dm_sel, preprocessing as dm_pre, impute as dm_imp, decomposition as dm_dec
from sklearn import inspection as sk_inspect, metrics as sk_metrics
from sklearn import model_selection as sk_sel, utils as sk_utils
from sklearn import pipeline
from sklearn.utils.multiclass import type_of_target

from hypernets.tabular import ToolBox, register_transformer
from hypernets.utils import logging, const, is_os_linux
from . import _collinearity, _drift_detection, _pseudo_labeling, _model_selection, _ensemble
from . import _data_cleaner
from . import _data_hasher
from . import _dataframe_mapper
from . import _feature_generators
from . import _metrics
from . import _transformers as tfs
from .. import sklearn_ex as sk_ex

try:
    import lightgbm

    lightgbm_installed = True
except ImportError:
    lightgbm_installed = False

logger = logging.get_logger(__name__)


def _reset_part_index(df, start):
    new_index = pd.RangeIndex.from_range(range(start, start + df.shape[0]))
    df.index = new_index
    return df


def _select_df_by_index(df, idx):
    return df[df.index.isin(idx)]


def _select_by_indices(part: pd.DataFrame, indices):
    return part.iloc[indices]


def _compute_chunk_sample_weight(y, classes, classes_weights):
    t = np.ones(y.shape[0])
    for i, c in enumerate(classes):
        t[y == c] *= classes_weights[i]
    return t


class DaskToolBox(ToolBox):
    acceptable_types = (dd.DataFrame, dd.Series, da.Array)
    compute = dask.compute

    @staticmethod
    def default_client():
        try:
            from dask.distributed import default_client as dask_default_client
            client = dask_default_client()
        except ValueError:
            client = None
        return client

    @staticmethod
    def dask_enabled():
        return DaskToolBox.default_client() is not None

    @staticmethod
    def is_local_dask():
        client = DaskToolBox.default_client()
        return type(client.cluster).__name__.lower().find('local') >= 0 if client is not None else False

    @staticmethod
    def dask_worker_count():
        client = DaskToolBox.default_client()
        return len(client.ncores()) if client else 0

    @staticmethod
    def dump_cluster_info(cluster=None, log_level=logging.INFO):
        if not logger.is_enabled_for(log_level):
            return

        if cluster is None:
            client = DaskToolBox.default_client()
            if client is None:
                logger.log(log_level, 'Not found dask default client.')
                return
            cluster = client.cluster

        msgs = [f'Dask cluster: {cluster}', ]
        try:
            msgs.append(f'scheduler: {cluster.scheduler}')
            msgs.append('workers:')
            GB = 1024 ** 3
            for i, wk in cluster.workers.items():
                mem_limit = wk.memory_limit / GB
                msgs.append(f'\t[{i}]: {wk}, mem: {mem_limit:.1f}GB')
        except:
            pass

        logger.log(log_level, '\n'.join(msgs))

    @staticmethod
    def is_dask_dataframe(X):
        return isinstance(X, dd.DataFrame)

    @staticmethod
    def is_dask_series(X):
        return isinstance(X, dd.Series)

    @staticmethod
    def is_dask_dataframe_or_series(X):
        return isinstance(X, (dd.DataFrame, dd.Series))

    @staticmethod
    def is_dask_array(X):
        return isinstance(X, da.Array)

    @staticmethod
    def is_dask_object(X):
        return isinstance(X, (da.Array, dd.DataFrame, dd.Series))

    @staticmethod
    def exist_dask_object(*args):
        for a in args:
            if isinstance(a, (da.Array, dd.DataFrame, dd.Series)):
                return True
            if isinstance(a, (tuple, list, set)):
                return DaskToolBox.exist_dask_object(*a)
        return False

    @staticmethod
    def exist_dask_dataframe(*args):
        for a in args:
            if isinstance(a, dd.DataFrame):
                return True
            if isinstance(a, (tuple, list, set)):
                return DaskToolBox.exist_dask_dataframe(*a)
        return False

    @staticmethod
    def exist_dask_array(*args):
        for a in args:
            if isinstance(a, da.Array):
                return True
            if isinstance(a, (tuple, list, set)):
                return DaskToolBox.exist_dask_array(*a)
        return False

    @staticmethod
    def to_dask_type(X):
        if isinstance(X, np.ndarray):
            worker_count = DaskToolBox.dask_worker_count()
            chunk_size = math.ceil(X.shape[0] / worker_count) if worker_count > 0 else X.shape[0]
            X = da.from_array(X, chunks=chunk_size)
        elif isinstance(X, (pd.DataFrame, pd.Series)):
            worker_count = DaskToolBox.dask_worker_count()
            partition_count = worker_count if worker_count > 0 else 1
            X = dd.from_pandas(X, npartitions=partition_count).clear_divisions()

        return X

    @staticmethod
    def to_dask_frame_or_series(X):
        X = DaskToolBox.to_dask_type(X)

        if isinstance(X, da.Array):
            X = dd.from_dask_array(X)

        return X

    @staticmethod
    def get_shape(X, allow_none=False):
        shape = ToolBox.get_shape(X, allow_none=allow_none)
        return dask.compute(shape)[0] if shape is not None else None

    @staticmethod
    def to_local(*data):
        return dask.compute(*data)

    @classmethod
    def from_local(cls, *data):
        return [cls.to_dask_type(t) for t in data]

    @staticmethod
    def load_data(data_path, *, reset_index=False, reader_mapping=None, **kwargs):
        import os.path as path
        import glob

        if reader_mapping is None:
            reader_mapping = {
                'csv': dd.read_csv,
                'txt': dd.read_csv,
                'parquet': dd.read_parquet,
                'par': dd.read_parquet,
                'json': dd.read_json,
            }

        if path.isdir(data_path) and not glob.has_magic(data_path):
            data_path = f'{data_path}*' if data_path.endswith(path.sep) else f'{data_path}{path.sep}*'

        df = ToolBox.load_data(data_path, reset_index=False, reader_mapping=reader_mapping, **kwargs)

        if reset_index:
            df = DaskToolBox.reset_index(df)

        worker_count = DaskToolBox.dask_worker_count()
        if worker_count > 1 and df.npartitions < worker_count:
            df = df.repartition(npartitions=worker_count)

        return df

    @staticmethod
    def unique(y):
        if isinstance(y, da.Array):
            uniques = da.unique(y).compute()
            uniques = set(uniques)
        elif isinstance(y, dd.Series):
            uniques = y.unique().compute()
            uniques = set(uniques)
        else:
            uniques = ToolBox.unique(y)
        return uniques

    @staticmethod
    def parquet():
        from . import _persistence
        return _persistence.DaskParquetPersistence()

    # @staticmethod
    # def unique_array(ar, return_index=False, return_inverse=False, return_counts=False, axis=None):
    #     assert axis is None or axis == 0
    #     return da.unique(ar, return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)
    @staticmethod
    def nunique_df(df):
        if isinstance(df, dd.DataFrame):
            columns = df.columns.to_list()
            uniques = [df[c].nunique() for c in columns]
            return {c: v for c, v in zip(columns, dask.compute(*uniques))}
        else:
            return ToolBox.nunique_df(df)

    @staticmethod
    def value_counts(ar):
        if isinstance(ar, da.Array):
            v_n = da.unique(ar, return_counts=True)
            v_n = dask.compute(*v_n)
            return {v: n for v, n in zip(*v_n)}
        elif isinstance(ar, dd.Series):
            s = ar
        elif isinstance(ar, dd.DataFrame):
            assert ar.shape[1] == 1
            s = ar.iloc[:, 0]
        else:
            return ToolBox.value_counts(ar)

        return s.value_counts().compute().to_dict()

    @staticmethod
    def reset_index(X):
        assert isinstance(X, (pd.DataFrame, dd.DataFrame))

        if DaskToolBox.is_dask_dataframe(X):
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

    @staticmethod
    def select_df(df, indices):
        """
        Select dataframe by row indices. For dask dataframe, call 'reset_index' before this.
        """
        assert isinstance(df, (pd.DataFrame, dd.DataFrame))

        if isinstance(df, dd.DataFrame):
            df = df.map_partitions(_select_df_by_index, indices, meta=df.dtypes.to_dict())
            return df
        else:
            return df.iloc[indices]

    @staticmethod
    def select_1d(arr, indices):
        """
        Select by indices from the first axis(0).
        """
        if isinstance(arr, (dd.DataFrame, dd.Series)):
            cache_attr = '_part_rows_'
            part_rows = None
            if hasattr(arr, cache_attr):
                part_rows = getattr(arr, cache_attr)

            if part_rows is None or len(part_rows) != arr.npartitions:
                part_rows = arr.map_partitions(lambda df: pd.DataFrame({'rows': [df.shape[0]]}),
                                               meta={'rows': 'int64'},
                                               ).compute()['rows'].values
                assert len(part_rows) == arr.npartitions
                setattr(arr, cache_attr, part_rows)

            part_indices = []
            indices = np.array(indices)
            for n, nc in zip(part_rows, np.cumsum(part_rows)):
                i_stop = nc
                i_start = nc - n
                idx = indices[indices >= i_start]  # filter indices
                idx = idx[idx < i_stop]  # filter indices
                idx = idx - i_start  # align to part internal
                part_indices.append(idx)

            delayed_reset_part_index = dask.delayed(_select_by_indices)
            parts = [delayed_reset_part_index(part, idx) for part, idx in zip(arr.to_delayed(), part_indices)]
            meta = arr.dtypes.to_dict() if isinstance(arr, dd.DataFrame) else (None, arr.dtype)
            X_new = dd.from_delayed(parts, prefix='ddf', meta=meta)
            return X_new
        else:
            return ToolBox.select_1d(arr, indices)

    @staticmethod
    def make_chunk_size_known(a):
        assert DaskToolBox.is_dask_array(a)

        chunks = a.chunks
        if any(np.nan in d for d in chunks):
            if logger.is_debug_enabled():
                logger.debug(f'call extracted array compute_chunk_sizes, shape: {a.shape}')
            a = a.compute_chunk_sizes()
        return a

    @staticmethod
    def make_divisions_known(X):
        assert DaskToolBox.is_dask_object(X)

        if DaskToolBox.is_dask_dataframe(X):
            if not X.known_divisions:
                # columns = X.columns.tolist()
                # X = X.reset_index()
                # new_columns = X.columns.tolist()
                # index_name = set(new_columns) - set(columns)
                # X = X.set_index(list(index_name)[0] if index_name else 'index')
                X = DaskToolBox.reset_index(X)
                assert X.known_divisions
        elif DaskToolBox.is_dask_series(X):
            if not X.known_divisions:
                name = X.name
                X = DaskToolBox.make_divisions_known(X.to_frame()).iloc[:, 0]
                X.name = name
        else:  # dask array
            X = DaskToolBox.make_chunk_size_known(X)

        return X

    @classmethod
    def hstack_array(cls, arrs):
        if all([a.ndim == 1 for a in arrs]):
            rows = dask.compute(arrs[0].shape)[0][0]
            arrs = [a.reshape(rows, 1) if a.ndim == 1 else a for a in arrs]
        return cls.stack_array(arrs, axis=1)

    @classmethod
    def vstack_array(cls, arrs):
        return cls.stack_array(arrs, axis=0)

    @staticmethod
    def stack_array(arrs, axis=0):
        assert axis in (0, 1)
        ndims = set([len(a.shape) for a in arrs])
        if len(ndims) > 1:
            assert ndims == {1, 2}
            assert all([len(a.shape) == 1 or a.shape[1] == 1 for a in arrs])
            arrs = [a.reshape(dask.compute(a.shape[0])[0], 1) if len(a.shape) == 1 else a for a in arrs]
        axis = min(axis, min([len(a.shape) for a in arrs]) - 1)
        assert axis >= 0

        if DaskToolBox.exist_dask_object(*arrs):
            arrs = [a.values if DaskToolBox.is_dask_dataframe_or_series(a) else a
                    for a in map(DaskToolBox.to_dask_type, arrs)]
            if len(arrs) > 1:
                arrs = [DaskToolBox.make_chunk_size_known(a) for a in arrs]
            return da.concatenate(arrs, axis=axis)
        else:
            return np.concatenate(arrs, axis=axis)

    @staticmethod
    def take_array(arr, indices, axis=None):
        if DaskToolBox.exist_dask_object(arr, indices):
            return da.take(arr, indices=indices, axis=axis)
        else:
            return np.take(arr, indices=indices, axis=axis)

    @staticmethod
    def array_to_df(arr, *, columns=None, index=None, meta=None):
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

        # convert array to dask Index object
        if isinstance(index, (np.ndarray, da.Array)):
            arr = DaskToolBox.make_chunk_size_known(arr)
            if isinstance(index, np.ndarray):
                index = da.from_array(index, chunks=arr.chunks[0])
            else:
                index = index.rechunk(arr.chunks[0])
            index = dd.from_dask_array(index).index

        df = dd.from_dask_array(arr, columns=columns, index=index, meta=meta)

        if isinstance(meta_df, (dd.DataFrame, pd.DataFrame)):
            dtypes_src = meta_df.dtypes
            dtypes_dst = df.dtypes
            for col in meta_df.columns:
                if dtypes_src[col] != dtypes_dst[col]:
                    df[col] = df[col].astype(dtypes_src[col])

        return df

    @staticmethod
    def df_to_array(df):
        if isinstance(df, dd.DataFrame):
            return df.to_dask_array(lengths=True)
        else:
            return ToolBox.df_to_array(df)

    @staticmethod
    def merge_oof(oofs):
        stacked = []
        for idx, proba in oofs:
            idx = idx.reshape(-1, 1)
            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)
            stacked.append(DaskToolBox.hstack_array([idx, proba]))
        df = dd.from_dask_array(DaskToolBox.vstack_array(stacked))
        df = df.set_index(0)
        r = df.to_dask_array(lengths=True)

        return r

    merge_oof.__doc__ = ToolBox.merge_oof.__doc__

    @staticmethod
    def select_valid_oof(y, oof):
        if isinstance(oof, da.Array):
            oof = DaskToolBox.make_chunk_size_known(oof)
            if len(oof.shape) == 1:
                nan_rows = da.isnan(oof[:])
            elif len(oof.shape) == 2:
                nan_rows = da.isnan(oof[:, 0])
            elif len(oof.shape) == 3:
                nan_rows = da.isnan(oof[:, 0, 0])
            else:
                raise ValueError(f'Unsupported shape:{oof.shape}')

            if nan_rows.sum().compute() == 0:
                return y, oof

            idx = da.argwhere(~nan_rows)
            idx = DaskToolBox.make_chunk_size_known(idx).ravel()
            idx = DaskToolBox.make_chunk_size_known(idx)
            if isinstance(y, da.Array):
                return y[idx], oof[idx]
            else:
                return DaskToolBox.select_1d(y, idx), oof[idx]
        else:
            return ToolBox.select_valid_oof(y, oof)

    @staticmethod
    def mean_oof(probas):
        if DaskToolBox.exist_dask_object(probas):
            probas = [DaskToolBox.df_to_array(p) if DaskToolBox.is_dask_dataframe_or_series(p) else p for p in probas]
            proba = probas[0]
            for i in range(1, len(probas)):
                proba += probas[i]
            proba = proba / len(probas)
        else:
            proba = ToolBox.mean_oof(probas)
        return proba

    @staticmethod
    def concat_df(dfs, axis=0, repartition=False, random_state=9527, **kwargs):
        if DaskToolBox.exist_dask_object(*dfs):
            dfs_orig = dfs
            dfs = [dd.from_dask_array(v) if DaskToolBox.is_dask_array(v) else v for v in dfs]

            if all([isinstance(df, (dd.Series, pd.Series)) for df in dfs]):
                values = DaskToolBox.vstack_array([df.values for df in dfs])
                df = dd.from_dask_array(values, columns=dfs[0].name)
                assert isinstance(df, dd.Series)
                return df

            if axis == 0:
                values = [df[dfs[0].columns].to_dask_array(lengths=True)
                          if not DaskToolBox.is_dask_array(df) else df for df in dfs_orig]
                df = DaskToolBox.array_to_df(DaskToolBox.vstack_array(values), meta=dfs[0])
            else:
                dfs = [DaskToolBox.make_divisions_known(df) for df in dfs]
                df = dd.concat(dfs, axis=axis, **kwargs)

            if DaskToolBox.is_dask_series(dfs[0]) and df.name is None and dfs[0].name is not None:
                df.name = dfs[0].name
            if repartition:
                df = df.shuffle(df.index, npartitions=dfs[0].npartitions)
        else:
            df = ToolBox.concat_df(dfs, axis=axis, repartition=repartition, random_state=random_state, **kwargs)

        return df

    @staticmethod
    def train_test_split(*data, shuffle=True, random_state=None, stratify=None, **kwargs):
        if DaskToolBox.exist_dask_dataframe(*data):
            if len(data) > 1:
                data = [DaskToolBox.make_divisions_known(DaskToolBox.to_dask_frame_or_series(x)) for x in data]
                head = data[0]
                for i in range(1, len(data)):
                    if data[i].divisions != head.divisions:
                        logger.info(f'repartition {i} from {data[i].divisions} to {head.divisions}')
                        data[i] = data[i].repartition(divisions=head.divisions)
            result = dm_sel.train_test_split(*data, shuffle=shuffle, random_state=random_state, **kwargs)
            result = [x.clear_divisions() for x in result]
        else:
            result = sk_sel.train_test_split(*data, shuffle=shuffle, random_state=random_state, stratify=stratify,
                                             **kwargs)

        return result

    @staticmethod
    def fix_binary_predict_proba_result(proba):
        if DaskToolBox.is_dask_object(proba):
            if proba.ndim == 1:
                proba = DaskToolBox.make_chunk_size_known(proba)
                proba = proba.reshape((proba.size, 1))
            if proba.shape[1] == 1:
                proba = DaskToolBox.hstack_array([1 - proba, proba])
        else:
            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            elif proba.shape[1] == 1:
                proba = np.hstack([1 - proba, proba])

        return proba

    @classmethod
    def general_estimator(cls, X, y=None, estimator=None, task=None):
        def default_dask_gbm(task):
            est_cls = lightgbm.dask.DaskLGBMRegressor if task == const.TASK_REGRESSION else lightgbm.dask.DaskLGBMClassifier
            return est_cls(n_estimators=50,
                           num_leaves=15,
                           max_depth=5,
                           subsample=0.5,
                           subsample_freq=1,
                           colsample_bytree=0.8,
                           reg_alpha=1,
                           reg_lambda=1,
                           importance_type='gain',
                           verbose=-1)

        if not cls.is_dask_object(X):
            return super().general_estimator(X, y, estimator=estimator, task=task)

        if (estimator is None or estimator == 'gbm') \
                and lightgbm_installed and hasattr(lightgbm, 'dask') \
                and is_os_linux:  # lightgbm.dask does not support windows
            return default_dask_gbm(task)

        estimator_ = super().general_estimator(X, y, estimator=estimator, task=task)
        estimator_ = cls.wrap_local_estimator(estimator_)
        return estimator_

    @staticmethod
    def wrap_for_local_scorer(estimator, target_type):
        def _call_and_compute(fn_call, fn_fix, *args, **kwargs):
            r = fn_call(*args, **kwargs)
            if DaskToolBox.is_dask_object(r):
                r = r.compute()
                if callable(fn_fix):
                    r = fn_fix(r)
            return r

        if hasattr(estimator, 'predict_proba'):
            orig_predict_proba = estimator.predict_proba
            fix = DaskToolBox.fix_binary_predict_proba_result if target_type == 'binary' else None
            setattr(estimator, '_orig_predict_proba', orig_predict_proba)
            setattr(estimator, 'predict_proba', partial(_call_and_compute, orig_predict_proba, fix))

        if hasattr(estimator, 'predict'):
            orig_predict = estimator.predict
            setattr(estimator, '_orig_predict', orig_predict)
            setattr(estimator, 'predict', partial(_call_and_compute, orig_predict, None))

        return estimator

    @staticmethod
    def compute_and_call(fn_call, *args, **kwargs):
        if logger.is_debug_enabled():
            logger.debug(f'[compute_and_call] compute {len(args)} object')

        args = dask.compute(*args, traverse=False)
        for k, v in kwargs.items():
            if DaskToolBox.exist_dask_object(v):
                kwargs[k] = dask.compute(v, traverse=True)[0]

        if logger.is_debug_enabled():
            logger.debug(f'[compute_and_call] call {fn_call.__name__}')
        # kwargs = {k: compute(v) if is_dask_array(v) else v for k, v in kwargs.items()}
        r = fn_call(*args, **kwargs)

        if logger.is_debug_enabled():
            logger.debug('[compute_and_call] to dask type')
        r = DaskToolBox.to_dask_type(r)

        if logger.is_debug_enabled():
            logger.debug('[compute_and_call] done')
        return r

    @staticmethod
    def call_and_compute(fn_call, optimize_graph, *args, **kwargs):
        if logger.is_debug_enabled():
            logger.debug(f'[call_and_compute] call {fn_call.__name__}')
        r = fn_call(*args, **kwargs)

        if DaskToolBox.is_dask_object(r):
            if logger.is_debug_enabled():
                logger.debug('[call_and_compute] to local type')
            r = dask.compute(r, traverse=False)[0]
        elif isinstance(r, (tuple, list)) and any(map(DaskToolBox.is_dask_object, r)):
            if logger.is_debug_enabled():
                logger.debug('[call_and_compute] to local type')
            # r = compute(*r, traverse=False, optimize_graph=optimize_graph)
            r = [x.compute() if DaskToolBox.is_dask_object(x) else x for x in r]

        if logger.is_debug_enabled():
            logger.debug('[call_and_compute] done')

        return r

    @staticmethod
    def wrap_local_estimator(estimator):
        for fn_name in ('fit', 'fit_cross_validation', 'predict', 'predict_proba'):
            fn_name_original = f'_wrapped_{fn_name}_by_wle'
            if hasattr(estimator, fn_name) and not hasattr(estimator, fn_name_original):
                fn = getattr(estimator, fn_name)
                assert callable(fn)
                setattr(estimator, fn_name_original, fn)
                setattr(estimator, fn_name, partial(DaskToolBox.compute_and_call, fn))

        return estimator

    @staticmethod
    def permutation_importance(estimator, X, y, *, scoring=None, n_repeats=5,
                               n_jobs=None, random_state=None):
        if not DaskToolBox.is_dask_dataframe(X):
            return sk_inspect.permutation_importance(estimator, X, y,
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

        if DaskToolBox.is_dask_object(y):
            y = y.compute()

        scorer = sk_metrics.check_scoring(DaskToolBox.wrap_for_local_scorer(estimator, type_of_target(y)), scoring)
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

    @staticmethod
    def compute_class_weight(class_weight, *, classes, y):
        if not DaskToolBox.is_dask_object(y):
            return sk_utils.class_weight.compute_class_weight(class_weight, classes=classes, y=y)

        y = DaskToolBox.make_chunk_size_known(y)
        if set(dask.compute(da.unique(y))[0]) - set(classes):
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
            y_shape, y_ind_bincount, le_classes_ = dask.compute(y.shape, da.bincount(y_ind), le.classes_)
            if not all(np.in1d(classes, le_classes_)):
                raise ValueError("classes should have valid labels that are in y")
            recip_freq = y_shape[0] / (len(le_classes_) * y_ind_bincount.astype(np.float64))
            weight = recip_freq[np.searchsorted(le_classes_, classes)]
        else:
            raise ValueError("Only class_weight == 'balanced' is supported.")

        return weight

    compute_class_weight.__doc__ = sk_utils.compute_class_weight.__doc__

    @staticmethod
    def compute_sample_weight(y):
        assert len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1)

        if hasattr(y, 'values'):
            y = y.values

        unique = dask.compute(da.unique(y))[0] if DaskToolBox.is_dask_object(y) else np.unique(y)
        cw = list(DaskToolBox.compute_class_weight('balanced', classes=unique, y=y))

        if DaskToolBox.is_dask_object(y):
            sample_weight = y.map_blocks(_compute_chunk_sample_weight, unique, cw, dtype=np.float64)
        else:
            sample_weight = _compute_chunk_sample_weight(y, unique, cw)

        return sample_weight

    # _data_cleaner_cls = data_cleaner_.DataCleaner
    _data_hasher_cls = _data_hasher.DaskDataHasher
    _data_cleaner_cls = _data_cleaner.DaskDataCleaner
    _collinearity_detector_cls = _collinearity.DaskMultiCollinearityDetector  # collinearity_.MultiCollinearityDetector
    _drift_detector_cls = _drift_detection.DaskDriftDetector  # drift_detection_.DriftDetector
    _feature_selector_with_drift_detection_cls = _drift_detection.DaskFeatureSelectionWithDriftDetector  # drift_detection_.FeatureSelectorWithDriftDetection
    _pseudo_labeling_cls = _pseudo_labeling.DaskPseudoLabeling  # pseudo_labeling_.PseudoLabeling
    _kfold_cls = _model_selection.FakeDaskKFold
    _stratified_kfold_cls = _model_selection.FakeDaskStratifiedKFold
    _greedy_ensemble_cls = _ensemble.DaskGreedyEnsemble
    metrics = _metrics.DaskMetrics


_predefined_transformers = dict(
    Pipeline=pipeline.Pipeline,
    ColumnTransformer=dm_comp.ColumnTransformer,
    SimpleImputer=dm_imp.SimpleImputer,
    StandardScaler=dm_pre.StandardScaler,
    MinMaxScaler=dm_pre.MinMaxScaler,
    RobustScaler=dm_pre.RobustScaler,
    # Normalizer=sk_pre.Normalizer,
    # KBinsDiscretizer=sk_pre.KBinsDiscretizer,
    LabelEncoder=dm_pre.LabelEncoder,
    OrdinalEncoder=dm_pre.OrdinalEncoder,
    OneHotEncoder=dm_pre.OneHotEncoder,
    PolynomialFeatures=dm_pre.PolynomialFeatures,
    QuantileTransformer=dm_pre.QuantileTransformer,
    # PowerTransformer=sk_pre.PowerTransformer,
    PCA=dm_dec.PCA,
    DataFrameMapper=_dataframe_mapper.DaskDataFrameMapper,
    PassThroughEstimator=sk_ex.PassThroughEstimator,

    AsTypeTransformer=sk_ex.AsTypeTransformer,
    # SafeLabelEncoder=sk_ex.SafeLabelEncoder,
    # LogStandardScaler=sk_ex.LogStandardScaler,
    # SkewnessKurtosisTransformer=sk_ex.SkewnessKurtosisTransformer,
    # FeatureSelectionTransformer=sk_ex.FeatureSelectionTransformer,
    # FloatOutputImputer=tfs.FloatOutputImputer,
    # DataFrameWrapper=,
    # GaussRankScaler=sk_ex.GaussRankScaler,
    # VarLenFeatureEncoder=tfs.VarLenFeatureEncoder,

    # MaxAbsScaler=tfs.MaxAbsScaler,
    # TruncatedSVD=tfs.TruncatedSVD,
    MultiLabelEncoder=tfs.SafeOrdinalEncoder,  # alias
    # SafeOrdinalEncoder=tfs.SafeOrdinalEncoder,
    # SafeOneHotEncoder=tfs.SafeOneHotEncoder,
    # LgbmLeavesEncoder=tfs.LgbmLeavesEncoder,
    # CategorizeEncoder=tfs.CategorizeEncoder,
    # MultiKBinsDiscretizer=tfs.MultiKBinsDiscretizer,
    # MultiVarLenFeatureEncoder=tfs.MultiVarLenFeatureEncoder,
    # LocalizedTfidfVectorizer=tfs.LocalizedTfidfVectorizer,

    # TfidfEncoder=sk_ex.TfidfEncoder,
    # DatetimeEncoder=sk_ex.DatetimeEncoder,
    # FeatureGenerationTransformer=_feature_generators.DaskFeatureGenerationTransformer,
    FeatureImportancesSelectionTransformer=sk_ex.FeatureImportancesSelectionTransformer,
)

if _feature_generators.is_feature_generator_ready:
    _predefined_transformers['FeatureGenerationTransformer'] = _feature_generators.DaskFeatureGenerationTransformer

for name, tf in _predefined_transformers.items():
    register_transformer(tf, name=name, dtypes=dd.DataFrame)
