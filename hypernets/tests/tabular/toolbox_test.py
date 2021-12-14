import pandas as pd

from hypernets.tabular import get_tool_box


class TestToolBox:
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
