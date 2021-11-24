# -*- coding:utf-8 -*-
"""

"""
from ._toolbox import DaskToolBox

try:
    import dask_ml.preprocessing as dm_pre
    import dask_ml.model_selection as dm_sel

    from dask_ml.impute import SimpleImputer
    from dask_ml.compose import ColumnTransformer
    from dask_ml.preprocessing import \
        LabelEncoder, OneHotEncoder, OrdinalEncoder, \
        StandardScaler, MinMaxScaler, RobustScaler

    from ._transformers import \
        SafeOneHotEncoder, TruncatedSVD, \
        MaxAbsScaler, SafeOrdinalEncoder, DataInterceptEncoder, \
        CallableAdapterEncoder, DataCacher, CacheCleaner, \
        LgbmLeavesEncoder, CategorizeEncoder, MultiKBinsDiscretizer, \
        LocalizedTfidfVectorizer, \
        MultiVarLenFeatureEncoder, DataFrameWrapper

    from ..sklearn_ex import PassThroughEstimator

    dask_ml_available = True
except ImportError:
    # Not found dask_ml
    dask_ml_available = False
