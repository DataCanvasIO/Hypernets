# -*- coding:utf-8 -*-

import time

from ..utils.common import config


def get_dispatcher(hyper_model, **kwargs):
    experiment = config('experiment', time.strftime('%Y%m%d%H%M%S'))
    work_dir = config('work-dir', f'experiments/{experiment}')

    if hyper_model.searcher.parallelizable:
        backend = config('search-backend', 'standalone')

        if backend == 'dask':
            from .dask.dask_dispatcher import DaskDispatcher
            return DaskDispatcher(config('models_dir', work_dir))
        elif config('role') is not None:
            role = config('role', 'standalone')
            driver_address = config('driver')
            if role == 'driver':
                from hypernets.dispatchers.cluster import DriverDispatcher
                return DriverDispatcher(driver_address, work_dir)
            elif role == 'executor':
                if driver_address is None:
                    raise Exception('Not found setting "driver" for executor role.')
                from hypernets.dispatchers.cluster import ExecutorDispatcher
                return ExecutorDispatcher(driver_address)

    return default_dispatcher(work_dir)


def default_dispatcher(work_dir=None):
    from .in_process_dispatcher import InProcessDispatcher

    models_dir = f'{work_dir}/models' if work_dir else ''
    return InProcessDispatcher(models_dir)
