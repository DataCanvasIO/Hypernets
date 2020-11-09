# -*- coding:utf-8 -*-

import time

from ..utils.common import config


def get_dispatcher(hyper_model, **kwargs):
    if hyper_model.searcher.parallelizable:
        dispatcher = config('dispatcher', 'standalone')
        if dispatcher == 'dask':
            from dask.distributed import Client, default_client
            try:
                default_client()
            except ValueError:
                # create default Client
                # client = Client("tcp://127.0.0.1:55208")
                # client = Client(processes=False, threads_per_worker=5, n_workers=1, memory_limit='4GB')
                Client()  # detect env: DASK_SCHEDULER_ADDRESS

            from .dask.dask_dispatcher import DaskDispatcher
            return DaskDispatcher()
        elif config('role') is not None:
            role = config('role', 'standalone')
            driver_address = config('driver')
            if role == 'driver':
                from hypernets.dispatchers.cluster import DriverDispatcher
                experiment = time.strftime('%Y%m%d%H%M%S')
                return DriverDispatcher(driver_address,
                                        config('spaces_dir', f'tmp/{experiment}/spaces'),
                                        config('spaces_dir', f'tmp/{experiment}/models'))
            elif role == 'executor':
                if driver_address is None:
                    raise Exception('Not found setting "driver" for executor role.')
                from hypernets.dispatchers.cluster import ExecutorDispatcher
                return ExecutorDispatcher(driver_address)

    return default_dispatcher()


def default_dispatcher():
    from .in_process_dispatcher import InProcessDispatcher
    return InProcessDispatcher()
