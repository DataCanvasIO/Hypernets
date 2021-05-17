import numpy as np

from hypernets.utils import infer_task_type, const


def test_infer_task_type():
    y1 = np.random.randint(0, 2, size=(1000), dtype='int')
    y2 = np.random.randint(0, 2, size=(1000)).astype('str')
    y3 = np.random.randint(0, 20, size=(1000)).astype('object')
    y4 = np.random.random(size=(1000)).astype('float')

    task, _ = infer_task_type(y1)
    assert (task, const.TASK_BINARY)

    task, _ = infer_task_type(y2)
    assert (task, const.TASK_BINARY)

    task, _ = infer_task_type(y3)
    assert (task, const.TASK_MULTICLASS)

    task, _ = infer_task_type(y4)
    assert (task, const.TASK_REGRESSION)
