# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import io

import numpy as np
import pandas as pd
from numpy import dtype

from hypernets.tabular import get_tool_box

csv_str = '''x1_int_nanchar,x2_all_nan,x3_const_str,x4_const_int,x5_dup_1,x6_dup_2,x7_dup_f1,x8_dup_f2,x9_f,x10,y
1.0,,const,5,dup,dup,0.1,0.1,1.23,\\N,1
2.2,,const,5,dupa,dupa,0.111,0.111,4.4,\\N,1
\\N,,const,5,dupb,dupb,0.12323,0.12323,1.233,\\N,1
4.,,const,5,dupc,dupc,0.14334,0.14334,4534434.2,\\N,0
5,,const,5,dupd,dupd,0.144,0.144,2302.2,\\N,0
6,,const,5,dupe,dupe,0.155,0.155,34334.1,\\N,\\N
'''


class TestDataCleaner:
    @classmethod
    def setup_class(cls):
        cls.df = cls.load_data()

    @staticmethod
    def load_data():
        return pd.read_csv(io.StringIO(csv_str))

    def test_basic(self):
        df = self.df.copy()
        tb = get_tool_box(df)
        print('clean', type(df), 'with', tb)
        # assert df.shape == (6, 11)
        assert df.shape[1] == 11
        assert list(df.dtypes.values) == [dtype('O'), dtype('float64'), dtype('O'), dtype('int64'), dtype('O'),
                                          dtype('O'), dtype('float64'), dtype('float64'), dtype('float64'),
                                          dtype('O'),
                                          dtype('O')]

        y = df.pop('y')
        cleaner = tb.data_cleaner(nan_chars='\\N',
                                  correct_object_dtype=True,
                                  drop_constant_columns=True,
                                  drop_label_nan_rows=True,
                                  drop_duplicated_columns=True,
                                  drop_idness_columns=False,
                                  replace_inf_values=np.nan
                                  )

        x_t, y_t = cleaner.fit_transform(df, y)
        x_t, y_t = tb.to_local(x_t, y_t)
        assert x_t.shape == (5, 4)
        assert y_t.shape == (5,)
        assert x_t.columns.to_list() == ['x1_int_nanchar', 'x5_dup_1', 'x7_dup_f1', 'x9_f']
        assert list(x_t.dtypes.values) == [dtype('float64'), dtype('O'), dtype('float64'), dtype('float64')]
        assert cleaner.df_meta_ == {'float64': ['x1_int_nanchar', 'x7_dup_f1', 'x9_f'], 'object': ['x5_dup_1']}

        cleaner.append_drop_columns(['x9_f'])

        assert cleaner.df_meta_ == {'float64': ['x1_int_nanchar', 'x7_dup_f1'], 'object': ['x5_dup_1']}
        x_t, y_t = cleaner.transform(df, y)
        x_t, y_t = tb.to_local(x_t, y_t)
        assert x_t.shape == (5, 3)
        assert y_t.shape == (5,)
        assert x_t.columns.to_list() == ['x1_int_nanchar', 'x5_dup_1', 'x7_dup_f1']
        assert list(x_t.dtypes.values) == [dtype('float64'), dtype('O'), dtype('float64')]

        cleaner = tb.data_cleaner(nan_chars='\\N',
                                  correct_object_dtype=True,
                                  drop_constant_columns=True,
                                  drop_label_nan_rows=True,
                                  drop_duplicated_columns=False,
                                  drop_idness_columns=False,
                                  replace_inf_values=np.nan
                                  )

        x_t, y_t = cleaner.fit_transform(df, y)
        x_t, y_t = tb.to_local(x_t, y_t)
        assert x_t.shape == (5, 6)
        assert y_t.shape == (5,)
        assert x_t.columns.to_list() == ['x1_int_nanchar', 'x5_dup_1', 'x6_dup_2', 'x7_dup_f1', 'x8_dup_f2', 'x9_f']
        assert list(x_t.dtypes.values) == [dtype('float64'), dtype('O'), dtype('O'), dtype('float64'),
                                           dtype('float64'),
                                           dtype('float64')]
        assert cleaner.df_meta_ == {'float64': ['x1_int_nanchar', 'x7_dup_f1', 'x8_dup_f2', 'x9_f'],
                                    'object': ['x5_dup_1', 'x6_dup_2']}

        cleaner = tb.data_cleaner(nan_chars='\\N',
                                  correct_object_dtype=True,
                                  drop_constant_columns=True,
                                  drop_label_nan_rows=False,
                                  drop_duplicated_columns=False,
                                  drop_idness_columns=False,
                                  replace_inf_values=np.nan
                                  )

        x_t, y_t = cleaner.fit_transform(df, y)
        x_t, y_t = tb.to_local(x_t, y_t)
        assert x_t.shape == (6, 6)
        assert y_t.shape == (6,)

        cleaner = tb.data_cleaner(nan_chars='\\N',
                                  correct_object_dtype=False,
                                  drop_constant_columns=True,
                                  drop_label_nan_rows=False,
                                  drop_duplicated_columns=False,
                                  drop_idness_columns=False,
                                  replace_inf_values=np.nan
                                  )

        x_t, y_t = cleaner.fit_transform(df, y)
        x_t, y_t = tb.to_local(x_t, y_t)
        assert x_t.shape == (6, 6)
        assert y_t.shape == (6,)
        assert x_t.columns.to_list() == ['x1_int_nanchar', 'x5_dup_1', 'x6_dup_2', 'x7_dup_f1', 'x8_dup_f2', 'x9_f']
        assert list(x_t.dtypes.values) == [dtype('O'), dtype('O'), dtype('O'), dtype('float64'), dtype('float64'),
                                           dtype('float64')]
        assert cleaner.df_meta_ == {'object': ['x1_int_nanchar', 'x5_dup_1', 'x6_dup_2'],
                                    'float64': ['x7_dup_f1', 'x8_dup_f2', 'x9_f']}

        cleaner = tb.data_cleaner(nan_chars='\\N',
                                  correct_object_dtype=False,
                                  drop_constant_columns=False,
                                  drop_label_nan_rows=False,
                                  drop_duplicated_columns=False,
                                  drop_idness_columns=False,
                                  replace_inf_values=np.nan
                                  )

        x_t, y_t = cleaner.fit_transform(df, y)
        x_t, y_t = tb.to_local(x_t, y_t)
        assert x_t.shape == (6, 10)
        assert y_t.shape == (6,)

        # not drop
        cleaner = tb.data_cleaner(nan_chars='\\N',
                                  correct_object_dtype=False,
                                  drop_constant_columns=True,
                                  drop_label_nan_rows=True,
                                  drop_duplicated_columns=True,
                                  drop_idness_columns=True,
                                  replace_inf_values=np.nan,
                                  reserve_columns=['x4_const_int']
                                  )

        x_t, y_t = cleaner.fit_transform(df, y)
        x_t, y_t = tb.to_local(x_t, y_t)
        assert 'x4_const_int' in x_t.columns.to_list()
