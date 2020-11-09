# -*- coding:utf-8 -*-

import time

from ..utils.common import config


def get_dispatcher(hyper_model, **kwargs):
    if hyper_model.searcher.parallelizable:
        experiment = config('experiment', time.strftime('%Y%m%d%H%M%S'))
        backend = config('search-backend', 'standalone')

        if backend == 'dask':
            from .dask.dask_dispatcher import DaskDispatcher
            return DaskDispatcher(config('models_dir', f'tmp/{experiment}/models'))
        elif config('role') is not None:
            role = config('role', 'standalone')
            driver_address = config('driver')
            if role == 'driver':
                from hypernets.dispatchers.cluster import DriverDispatcher
                return DriverDispatcher(driver_address,
                                        config('spaces_dir', f'tmp/{experiment}/spaces'),
                                        config('models_dir', f'tmp/{experiment}/models'))
            elif role == 'executor':
                if driver_address is None:
                    raise Exception('Not found setting "driver" for executor role.')
                from hypernets.dispatchers.cluster import ExecutorDispatcher
                return ExecutorDispatcher(driver_address)

    return default_dispatcher()


def default_dispatcher():
    from .in_process_dispatcher import InProcessDispatcher
    return InProcessDispatcher()
