# -*- coding:utf-8 -*-
"""

"""
import pytest

from hypernets.tabular import is_cuml_installed

if is_cuml_installed:
    import cupy

    if_cuml_ready = pytest.mark.skipif(not cupy.cuda.is_available(), reason='Cuda is not available')
else:
    if_cuml_ready = pytest.mark.skipif(not is_cuml_installed, reason='Cuml is not installed')
