import os.path as path

import numpy as np
import pandas as pd

from hypernets.tabular import get_tool_box
from hypernets.tabular.datasets import dsutils
from hypernets.utils import const


class TestToolBox:

    def test_infer_task_type(self):
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

    def test_detect_estimator_lightgbm(self):
        tb = get_tool_box(pd.DataFrame)
        detector = tb.estimator_detector('lightgbm.LGBMClassifier', 'binary')
        r = detector()
        assert r == {'installed', 'initialized', 'fitted'}

    def test_concat_df(self):
        df = pd.DataFrame(dict(
            x1=['a', 'b', 'c'],
            x2=[1, 2, 3]
        ))
        tb = get_tool_box(pd.DataFrame)

        # DataFrame + DataFrame
        df1 = tb.concat_df([df, df], axis=0)
        df2 = pd.concat([df, df], axis=0)
        assert (df1 == df2).all().all()

        # DataFrame + ndarray
        df1 = tb.concat_df([df, df.values], axis=0)
        df2 = pd.concat([df, df], axis=0)
        assert isinstance(df1, pd.DataFrame)
        assert (df1 == df2).all().all()

        # Series + ndarray
        s = df['x1']
        df1 = tb.concat_df([s, s.values], axis=0)
        df2 = pd.concat([s, s], axis=0)
        assert isinstance(df1, pd.Series)
        assert (df1 == df2).all()

    def test_load_data(self, ):
        data_dir = path.split(dsutils.__file__)[0]
        data_file = f'{data_dir}/blood.csv'
        tb = get_tool_box(pd.DataFrame)
        df = tb.load_data(data_file)
        assert isinstance(df, pd.DataFrame)
