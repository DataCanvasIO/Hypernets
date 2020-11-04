# -*- coding:utf-8 -*-

from ..utils.common import config


def get_dispatcher(hyper_model, **kwargs):
    if hyper_model.searcher.parallelizable:
        cluster = config('cluster', 'standalone')
        if cluster == 'dask':
            from dask.distributed import Client
            # client = Client("tcp://127.0.0.1:55208")
            # client = Client(processes=False, threads_per_worker=5, n_workers=1, memory_limit='4GB')
            Client()  # detect env: DASK_SCHEDULER_ADDRESS

            from .dask_dispatcher import DaskDispatcher
            return DaskDispatcher()

        role = config('role', 'standalone')
        driver_address = config('driver')
        if role == 'driver':
            from .driver_dispatcher import DriverDispatcher
            return DriverDispatcher(driver_address, config('spaces_dir', 'spaces'))
        elif role == 'executor':
            if driver_address is None:
                raise Exception('Not found setting "driver" for executor role.')
            from .executor_dispatcher import ExecutorDispatcher
            return ExecutorDispatcher(driver_address)

    return default_dispatcher()


def default_dispatcher():
    from .in_process_dispatcher import InProcessDispatcher
    return InProcessDispatcher()
