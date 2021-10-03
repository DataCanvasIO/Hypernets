# -*- coding:utf-8 -*-
"""

"""

from .toolbox import ToolBox

TOOL_BOXES = [ToolBox]


def get_tool_box(*data):
    for tb in TOOL_BOXES:
        if tb.accept(*data):
            return tb

    raise ValueError(f'No toolbox found for your data with types: {[type(x) for x in data]}. '
                     f'Registered tabular toolboxes are {[t.__name__ for t in TOOL_BOXES]}.')


try:
    import dask.dataframe as dd

    import dask_ml
    from .dask_ex import DaskToolBox

    TOOL_BOXES.insert(0, DaskToolBox)
    is_dask_installed = True
except ImportError:
    # import traceback
    # traceback.print_exc()
    is_dask_installed = False

# print(f'Registered tabular toolboxes are {[t.__name__ for t in TOOL_BOXES]}.')
