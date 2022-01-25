# -*- coding:utf-8 -*-
"""

"""
import math
import os

import psutil
import pytest

from hypernets.tabular import is_dask_installed

if_dask_ready = pytest.mark.skipif(not is_dask_installed, reason='dask or dask_ml are not installed')


def _startup_dask(overload):
    from dask.distributed import LocalCluster, Client

    if os.environ.get('DASK_SCHEDULER_ADDRESS') is not None:
        # use dask default settings
        client = Client()
    else:
        # start local cluster
        cores = psutil.cpu_count()
        workers = math.ceil(cores / 3)
        workers = max(2, workers)
        if workers > 1:
            if overload <= 0:
                overload = 1.0
            mem_total = psutil.virtual_memory().available / (1024 ** 3)  # GB
            mem_per_worker = math.ceil(mem_total / workers * overload)
            if mem_per_worker > mem_total:
                mem_per_worker = mem_total
            cluster = LocalCluster(processes=True, n_workers=workers, threads_per_worker=4,
                                   memory_limit=f'{mem_per_worker}GB')
        else:
            cluster = LocalCluster(processes=False)

        client = Client(cluster)
    return client


def setup_dask(cls):
    try:
        from dask.distributed import default_client
        client = default_client()
    except:
        client = _startup_dask(2.0)
    print('Dask Client:', client)

    if cls is not None:
        setattr(cls, 'dask_client_', client)
