# -*- coding:utf-8 -*-
"""

"""
from ._base import get_tool_box, register_toolbox, register_transformer, tb_transformer
from .toolbox import ToolBox

register_toolbox(ToolBox, aliases=('default', 'pandas'))

try:
    import dask.dataframe as dd

    import dask_ml
    from .dask_ex import DaskToolBox

    register_toolbox(DaskToolBox, pos=0, aliases=('dask',))
    is_dask_installed = True
except ImportError:
    # import traceback
    # traceback.print_exc()
    is_dask_installed = False

try:
    import cupy
    import cudf
    import cuml
    from .cuml_ex import CumlToolBox

    register_toolbox(CumlToolBox, pos=0, aliases=('cuml', 'rapids'))
    is_cuml_installed = True
except ImportError:
    # import traceback
    #
    # traceback.print_exc()
    is_cuml_installed = False
