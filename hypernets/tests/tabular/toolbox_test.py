from hypernets.tabular import get_tool_box
import pandas as pd


class TestToolBox:
    def test_detect_estimator_lightgbm(self):
        tb = get_tool_box(pd.DataFrame)
        detector = tb.estimator_detector('lightgbm.LGBMClassifier', 'binary')
        r = detector()
        assert r == {'installed', 'initialized', 'fitted'}
