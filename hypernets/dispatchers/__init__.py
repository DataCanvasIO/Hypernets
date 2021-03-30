# -*- coding:utf-8 -*-

import time

from .cfg import DispatchCfg as c


def get_dispatcher(hyper_model, **kwargs):
    timestamp = time.strftime('%Y%m%d%H%M%S')
    experiment = c.experiment if len(c.experiment) > 0 else f'experiment_{timestamp}'
    work_dir = c.work_dir if len(c.work_dir) > 0 else f'{experiment}'

    if hyper_model.searcher.parallelizable:
        if c.backend == 'dask':
            from .dask.dask_dispatcher import DaskDispatcher
            return DaskDispatcher(work_dir)
        elif c.backend == 'cluster':
            driver_address = c.cluster_driver
            if c.cluster_role == 'driver':
                from hypernets.dispatchers.cluster import DriverDispatcher
                return DriverDispatcher(driver_address, work_dir)
            elif c.cluster_role == 'executor':
                if driver_address is None:
                    raise Exception('Not found setting "driver" for executor role.')
                from hypernets.dispatchers.cluster import ExecutorDispatcher
                return ExecutorDispatcher(driver_address)

    return default_dispatcher(work_dir)


def default_dispatcher(work_dir=None):
    from .in_process_dispatcher import InProcessDispatcher

    models_dir = f'{work_dir}/models' if work_dir else ''
    return InProcessDispatcher(models_dir)
