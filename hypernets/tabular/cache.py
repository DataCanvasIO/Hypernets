import inspect
import pickle
from functools import partial

import pandas as pd
from sklearn.base import BaseEstimator

from hypernets import __version__
from hypernets.tabular import get_tool_box
from hypernets.utils import fs, logging
from .cfg import TabularCfg as cfg

logger = logging.get_logger(__name__)

_STRATEGY_DATA = 'data'
_STRATEGY_TRANSFORM = 'transform'

_KIND_DEFAULT = 'pickle'
_KIND_PARQUET = 'parquet'
_KIND_LIST = 'list'
_KIND_NONE = 'none'

_is_parquet_ready_flag = None


def _is_parquet_ready(tb):
    global _is_parquet_ready_flag
    if _is_parquet_ready_flag is None:
        try:
            tb.parquet()
            _is_parquet_ready_flag = True
        except ImportError as e:
            logger.warning(f'{e}, so cache strategy "{_STRATEGY_DATA}" is disabled.')
            _is_parquet_ready_flag = False
    return _is_parquet_ready_flag


class SkipCache(Exception):
    pass


class CacheCallback:
    def on_enter(self, fn, *args, **kwargs):
        """
        is fired before checking cache.
        raise Exception to disable cache
        """
        pass

    def on_apply(self, fn, cached_data, *args, **kwargs):
        """
        is fired before applying cached data.
        raise Exception to skip applying
        """
        pass

    def on_store(self, fn, cached_data, *args, **kwargs):
        """
        is fired before storing cache.
        raise Exception to skip store cache
        """
        pass

    def on_leave(self, fn, *args, **kwargs):
        """
        is fired before leaving fn call.
        raise Exception to skip store cache
        """
        pass


def cache(strategy=None, arg_keys=None, attr_keys=None, attrs_to_restore=None, transformer=None,
          callbacks=None, cache_dir=None):
    assert strategy in [_STRATEGY_TRANSFORM, _STRATEGY_DATA, None]
    assert isinstance(arg_keys, (tuple, list, str, type(None)))
    assert isinstance(attr_keys, (tuple, list, str, type(None)))
    assert isinstance(attrs_to_restore, (tuple, list, str, type(None)))
    assert callable(transformer) or isinstance(transformer, str) or transformer is None
    assert callbacks is None or isinstance(callbacks, CacheCallback) \
           or all([issubclass(type(c), CacheCallback) for c in callbacks])

    if isinstance(arg_keys, str):
        arg_keys = [a.strip(' ') for a in arg_keys.split(',') if len(a.strip(' ')) > 0]
    if isinstance(attr_keys, str):
        attr_keys = [a.strip(' ') for a in attr_keys.split(',') if len(a.strip(' ')) > 0]
    if isinstance(attrs_to_restore, str):
        attrs_to_restore = [a.strip(' ') for a in attrs_to_restore.split(',') if len(a.strip(' ')) > 0]

    if isinstance(callbacks, CacheCallback):
        callbacks = [callbacks]

    return partial(decorate,
                   strategy=strategy,
                   cache_dir=cache_dir,
                   attr_keys=attr_keys,
                   arg_keys=arg_keys,
                   attrs_to_restore=attrs_to_restore,
                   transformer=transformer,
                   callbacks=callbacks)


def decorate(fn, *, cache_dir, strategy,
             arg_keys=None, attr_keys=None, attrs_to_restore=None,
             transformer=None, callbacks=None):
    assert callable(fn)

    sig = inspect.signature(fn)
    if isinstance(transformer, str) or attr_keys is not None or attrs_to_restore is not None:
        assert 'self' in sig.parameters.keys()

    if cfg.cache_strategy == 'disabled':
        return fn

    if callbacks is None:
        callbacks = []

    if cache_dir is None:
        cache_dir = f'{cfg.cache_dir}{fs.sep}{".".join([fn.__module__, fn.__qualname__])}'

    if cfg.cache_strategy != 'disabled' and not fs.exists(cache_dir):
        try:
            fs.mkdirs(cache_dir, exist_ok=True)
        except:
            logger.warning(f'Failed to create cache directory "{cache_dir}".')

    def _cache_call(*args, **kwargs):
        assert len(args) > 0

        obj = None
        cache_path = None
        loaded = False
        result = None
        tb = _get_tool_box_for_cache(*args, **kwargs)

        try:
            for c in callbacks:
                c.on_enter(fn, *args, **kwargs)

            # bind arguments
            bind_args = sig.bind(*args, **kwargs)
            bind_args.apply_defaults()

            obj = bind_args.arguments.get('self', None)

            # calc cache_key
            key_items = {}

            arg_kwargs = bind_args.arguments.get('kwargs', {}).copy()
            arg_items = {k: v for k, v in bind_args.arguments.items() if k not in ['self', ]}  # as dict
            arg_items.update(arg_kwargs)

            if arg_keys is not None and len(arg_keys) > 0:
                key_items.update({k: arg_items.get(k) for k in arg_keys})
            else:
                key_items.update(arg_items)

            if attr_keys is not None:
                key_items.update({k: getattr(obj, k, None) for k in attr_keys})
            elif isinstance(obj, BaseEstimator) and 'params_' not in key_items:
                key_items['params_'] = obj.get_params(deep=False)

            if attrs_to_restore is not None:
                key_items['attrs_to_restore_'] = attrs_to_restore

            cache_key = tb.data_hasher()(key_items)

            # join cache_path
            if not fs.exists(cache_dir):
                fs.mkdirs(cache_dir, exist_ok=True)
            cache_path = f'{cache_dir}{fs.sep}{cache_key}'

            # detect and load cache
            if fs.exists(f'{cache_path}.meta'):
                # load
                cached_data, meta = _load_cache(tb, cache_path)

                for c in callbacks:
                    c.on_apply(fn, cached_data, *args, **kwargs)

                # restore attributes
                original_attributes = {}
                if attrs_to_restore is not None:
                    cached_attributes = meta.get('attributes', {})
                    for k in attrs_to_restore:
                        original_attributes[k] = getattr(obj, k, None)
                        setattr(obj, k, cached_attributes.get(k))

                if meta['strategy'] == _STRATEGY_DATA:
                    result = cached_data
                    loaded = True
                else:  # strategy==transform
                    try:
                        if isinstance(transformer, str):
                            tfn = getattr(obj, transformer)
                            assert callable(tfn)
                            result = tfn(*args[1:], **kwargs)  # exclude args[0]==self
                        elif callable(transformer):
                            result = transformer(*args, **kwargs)
                        loaded = True
                    except:
                        # unexpected error, restore original attrs
                        logger.warning(f'Failed to transform {type(obj).__name__}, '
                                       f'the dependencies may have been changed, so try to disable cache.')
                        for k, a in original_attributes.items():
                            setattr(obj, k, a)
                        loaded = False
        except SkipCache:
            pass
        except Exception as e:
            logger.warning(e)

        if not loaded:
            result = fn(*args, **kwargs)

        if cache_path is not None and not loaded:
            try:
                for c in callbacks:
                    c.on_store(fn, result, *args, **kwargs)

                # store cache
                cache_strategy = strategy if strategy is not None else cfg.cache_strategy
                if cache_strategy == _STRATEGY_DATA and not _is_parquet_ready(tb):
                    cache_strategy = _STRATEGY_TRANSFORM

                if cache_strategy == _STRATEGY_TRANSFORM and (result is None or transformer is not None):
                    cache_data = None
                    meta = {'strategy': _STRATEGY_TRANSFORM}
                else:
                    cache_data = result
                    meta = {'strategy': _STRATEGY_DATA}

                if attrs_to_restore is not None:
                    meta['attributes'] = {k: getattr(obj, k, None) for k in attrs_to_restore}
                if isinstance(obj, BaseEstimator):
                    meta['params_'] = obj.get_params(deep=False)  # for info

                _store_cache(tb, cache_path, cache_data, meta=meta)

                for c in callbacks:
                    c.on_leave(fn, *args, **kwargs)

            except Exception as e:
                logger.warning(e)

        return result

    return _cache_call


def _get_tool_box_for_cache(*args, **kwargs):
    dtypes = []
    for a in args:
        stype = str(type(a))
        if stype.find('DataFrame') >= 0 or stype.find('array') >= 0 or stype.find('Array') >= 0:
            dtypes.append(type(a))
    if len(dtypes) == 0:
        dtypes.append(pd.DataFrame)

    return get_tool_box(*dtypes)


def _store_cache(toolbox, cache_path, data, meta):
    meta = meta.copy() if meta is not None else {}
    meta['version'] = __version__

    if data is None:
        meta.update({'kind': _KIND_NONE, 'items': []})
    elif isinstance(data, (list, tuple)):
        items = [f'_{i}' for i in range(len(data))]
        for d, i in zip(data, items):
            _store_cache(toolbox, f'{cache_path}{i}', d, meta)
        meta.update({'kind': _KIND_LIST, 'items': items})
    else:
        pq = toolbox.parquet()
        if isinstance(data, pq.acceptable_types):
            item = f'.parquet'
            pq.store(data, f'{cache_path}{item}', filesystem=fs)
            meta.update({'kind': _KIND_PARQUET, 'items': [item]})
        else:
            item = f'.pkl'
            with fs.open(f'{cache_path}{item}', 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            meta.update({'kind': _KIND_DEFAULT, 'items': [item]})

    with fs.open(f'{cache_path}.meta', 'wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_cache(toolbox, cache_path):
    with fs.open(f'{cache_path}.meta', 'rb') as f:
        meta = pickle.load(f)

    if meta['version'] != __version__:
        raise EnvironmentError(f'Incompatible version: {meta["version"]}, please clear cache and try again.')

    data_kind = meta['kind']
    items = meta['items']

    if data_kind == _KIND_NONE:
        data = None
    elif data_kind == _KIND_LIST:
        data = [_load_cache(toolbox, f'{cache_path}{i}')[0] for i in items]
    elif data_kind == _KIND_DEFAULT:  # pickle
        with fs.open(f'{cache_path}{items[0]}', 'rb') as f:
            data = pickle.load(f)
    elif data_kind == _KIND_PARQUET:
        pq = toolbox.parquet()
        data = pq.load(f'{cache_path}{items[0]}', filesystem=fs)
    else:
        raise ValueError(f'Unexpected cache data kind "{data_kind}"')

    return data, meta


def clear(cache_dir=None, fn=None):
    assert fn is None or callable(fn)

    if cache_dir is None:
        cache_dir = cfg.cache_dir
    if callable(fn):
        cache_dir = f'{cache_dir}{fs.sep}{".".join([fn.__module__, fn.__qualname__])}'

    if fs.exists(cache_dir):
        fs.rm(cache_dir, recursive=True)
        fs.mkdirs(cache_dir, exist_ok=True)
