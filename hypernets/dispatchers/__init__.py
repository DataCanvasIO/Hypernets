# -*- coding:utf-8 -*-

from ..utils.common import config


def get_dispatcher(hyper_model, **kwargs):
    if hyper_model.searcher.parallelizable:
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
