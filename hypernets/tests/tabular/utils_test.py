# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import copy
import io

import pandas as pd

from hypernets.tabular import get_tool_box

csv_str = '''x1_int_nanchar,x2_all_nan,x3_const_str,x4_const_int,x5_dup_1,x6_dup_2,x7_dup_f1,x8_dup_f2,x9_f,x10,y
1.0,,const,5,dup,dup,0.1,0.1,1.23,\\N,1
2.2,,const,5,dupa,dupa,0.111,0.111,4.4,\\N,1
\\N,,const,5,dupb,dupb,0.12323,0.12323,1.233,\\N,1
4.,,const,5,dupc,dupc,0.14334,0.14334,4534434.2,\\N,0
5,,const,5,dupd,dupd,0.144,0.144,2302.2,\\N,0
6,,const,5,dupe,dupe,0.155,0.155,34334.1,\\N,\\N
'''


class Test_DataCleaner():
    def test_basic(self):
        hasher = get_tool_box(pd.DataFrame).data_hasher()
        df1 = pd.read_csv(io.StringIO(csv_str))
        hash1 = hasher(df1)

        df2 = pd.read_csv(io.StringIO(csv_str))
        hash2 = hasher(df2)
        assert hash1 == hash2

        df3 = df1.head(5)
        hash3 = hasher(df3)
        assert hash1 != hash3

        df4 = pd.concat([df1, df1.head(1)], axis=0)
        hash4 = hasher(df4)
        assert hash1 != hash4

        df5 = copy.deepcopy(df1)
        df5['x1_int_nanchar'] = ['1.0', '2.2', '\\N', '4.', '5', '6']
        hash5 = hasher(df5)
        assert hash1 == hash5

        df6 = copy.deepcopy(df1)
        df6['x1_int_nanchar'] = ['2.0', '2.2', '\\N', '4.', '5', '6']
        hash6 = hasher(df6)
        assert hash1 != hash6

    # TODO @lxf add unit tests for Dask.DataFrame
