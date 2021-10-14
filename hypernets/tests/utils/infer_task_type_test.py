import numpy as np

from hypernets.tabular import get_tool_box
from hypernets.utils import const


def test_infer_task_type():
    y1 = np.random.randint(0, 2, size=(1000), dtype='int')
    y2 = np.random.randint(0, 2, size=(1000)).astype('str')
    y3 = np.random.randint(0, 20, size=(1000)).astype('object')
    y4 = np.random.random(size=(1000)).astype('float')
    y5 = np.array([1, 1, 2, 2, 'na'])

    tb = get_tool_box(y1)

    task, _ = tb.infer_task_type(y1)
    assert task == const.TASK_BINARY

    task, _ = tb.infer_task_type(y2)
    assert task == const.TASK_BINARY

    task, _ = tb.infer_task_type(y3)
    assert task == const.TASK_MULTICLASS

    task, _ = tb.infer_task_type(y4)
    assert task == const.TASK_REGRESSION

    task, _ = tb.infer_task_type(y5, excludes=['na'])
    assert task == const.TASK_BINARY
