# -*- coding:utf-8 -*-
import argparse


def get_dispatcher(hyper_model, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', '-role',
                        default='standalone', choices=['driver', 'executor', 'standalone'],
                        help='process role, one of driver/executor/standalone')
    parser.add_argument('--driver', '-driver',
                        help='address and port of the driver process, with format: "<hostname>:<port>"')
    parser.add_argument('--spaces-dir', '-spaces-dir',
                        default='spaces',
                        help='[driver only] director to store space sample file, default "spaces"')
    args, _ = parser.parse_known_args()

    if hyper_model.searcher.parallelizable:
        role = args.role
        driver_address = args.driver
        if role == 'driver':
            from .driver_dispatcher import DriverDispatcher
            return DriverDispatcher(driver_address, args.spaces_dir)
        elif role == 'executor':
            from .executor_dispatcher import ExecutorDispatcher
            return ExecutorDispatcher(driver_address)

    return default_dispatcher()


def default_dispatcher():
    from .in_process_dispatcher import InProcessDispatcher
    return InProcessDispatcher()
