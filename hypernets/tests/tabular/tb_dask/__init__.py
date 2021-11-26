# -*- coding:utf-8 -*-
"""

"""
import pytest

from hypernets.tabular import is_dask_installed

if_dask_ready = pytest.mark.skipif(not is_dask_installed, reason='dask or dask_ml are not installed')
